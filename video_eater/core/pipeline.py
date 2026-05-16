# pipeline.py
import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

from video_eater.core.handle_video.audio_extractor import AudioExtractor
from video_eater.core.config_models import VideoProject, ProcessingStats, ProcessingConfig
from video_eater.core.output_templates import (
    YouTubeDescriptionFormatter,
    MarkdownReportFormatter,
    JsonFormatter,
    SimpleTextFormatter,
    PlainTextTranscriptFormatter,
    SrtTranscriptFormatter,
    MarkdownTranscriptFormatter,
)
from video_eater.core.transcribe_audio.transcribe_audio_chunks import transcribe_audio_chunk_folder
from video_eater.core.ai_processors.transcript_processor import TranscriptProcessor
from video_eater.core.ai_processors.ai_prompt_models import FullVideoAnalysis
from video_eater.core.transcribe_audio.transcript_models import VideoTranscript

logger = logging.getLogger(__name__)


class VideoProcessingPipeline:
    """Clean, maintainable video processing pipeline."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.stats = ProcessingStats()

    async def process_video(self, project: VideoProject) -> "PipelineResult":
        """Process a single video through the pipeline."""

        logger.info(f"Processing: {project.video_path.name}")

        # Each step is now clean and focused
        audio_chunks = await self._extract_audio(project)
        transcripts = await self._transcribe_chunks(project)
        analysis = await self._analyze_transcripts(project, transcripts)
        _ = await self._generate_outputs(project, analysis, transcripts)
        return PipelineResult(
            project=project,
            stats=self.stats,
            analysis=analysis
        )

    async def _extract_audio(self, project: VideoProject) -> list[Path]:
        """Extract audio with clean separation of concerns."""

        extractor = AudioExtractor(
            chunk_length=self.config.chunk_length_seconds,
            overlap=self.config.chunk_overlap_seconds
        )

        # Check cache
        if not self.config.force_chunk_audio:
            existing = extractor.find_existing_chunks(project.audio_chunks_folder)
            if existing:
                logger.info(f"Using {len(existing)} cached audio chunks")
                self.stats.audio_chunks_cached = len(existing)
                return existing

        # Extract new chunks
        chunks = await extractor.extract(
            video_path=project.video_path,
            output_folder=project.audio_chunks_folder
        )

        self.stats.audio_chunks_created = len(chunks)
        logger.success(f"Created {len(chunks)} audio chunks")

        return chunks

    async def _transcribe_chunks(self, project: VideoProject) -> list[VideoTranscript]:
        """Transcribe audio chunks and return transcripts."""
        project.transcript_chunks_folder.mkdir(parents=True, exist_ok=True)

        before_count = len(list(project.transcript_chunks_folder.glob("*.transcript.json")))

        whisper_prompt = None
        if self.config.whisper_vocabulary:
            whisper_prompt = "Expected terms: " + ", ".join(self.config.whisper_vocabulary)

        transcripts = await transcribe_audio_chunk_folder(
            audio_chunk_folder=str(project.audio_chunks_folder),
            transcript_chunk_folder=str(project.transcript_chunks_folder),
            file_extension=".mp3",
            re_transcribe=self.config.force_transcribe,
            whisper_prompt=whisper_prompt,
        )

        after_count = len(list(project.transcript_chunks_folder.glob("*.transcript.json")))
        created = max(0, after_count - before_count) if not self.config.force_transcribe else after_count
        cached = 0 if self.config.force_transcribe else before_count
        self.stats.transcripts_created = created
        self.stats.transcripts_cached = cached
        logger.success(f"Prepared {after_count} transcript chunks ({created} new, {cached} cached)")
        return transcripts

    async def _analyze_transcripts(
        self,
        project: VideoProject,
        transcripts: list[VideoTranscript],
    ) -> FullVideoAnalysis:
        """Analyze transcripts into a full video analysis."""
        processor = TranscriptProcessor(
            model=self.config.analysis_model,
            use_async=True,
            max_concurrent_chunks=self.config.max_concurrent_chunks,
            batch_size=self.config.batch_size,
            chunk_length_seconds=self.config.chunk_length_seconds,
        )

        chunk_analyses = await processor.process_transcript_folder(
            transcript_folder=project.transcript_chunks_folder,
            chunk_analysis_output_folder=project.analysis_folder,
        )
        # Wire per-chunk stats back into pipeline stats
        self.stats.analyses_created = processor.processing_stats['chunks_processed']
        self.stats.analyses_cached = processor.processing_stats['chunks_cached']

        # Combine all analyses (or load from cache if already done)
        combined_file = project.output_folder / f"{project.video_path.stem}_full_video_analysis.yaml"

        if combined_file.exists() and not self.config.force_analyze:
            logger.info(f"Using cached full video analysis from {combined_file}")
            with open(combined_file, 'r', encoding='utf-8') as f:
                full_analysis = FullVideoAnalysis(**yaml.safe_load(f))
            await processor.close()
            return full_analysis

        logger.info("Combining all chunk analyses...")
        full_analysis = await processor.combine_analyses(chunk_analyses)
        await processor.close()

        # Save combined analysis as YAML
        with open(combined_file, 'w', encoding='utf-8') as f:
            yaml.dump(full_analysis.model_dump(), f,
                      default_flow_style=False,
                      sort_keys=False,
                      allow_unicode=True)
        logger.success(f"Saved combined analysis to {combined_file}")

        return full_analysis

    async def _generate_outputs(
        self,
        project: VideoProject,
        analysis: FullVideoAnalysis,
        transcripts: list[VideoTranscript],
    ) -> list[str]:
        """Generate outputs using configurable formatters."""

        output_folder = project.output_folder
        output_folder.mkdir(parents=True, exist_ok=True)

        stem = project.video_path.stem
        generated = []

        # Analysis-based outputs: (filename, formatter, max_length_or_None)
        analysis_outputs: list[tuple[str, object, int | None]] = [
            (f'{stem}_youtube_description_full.md',      YouTubeDescriptionFormatter(), None),
            (f'{stem}_youtube_description_truncated.md',  YouTubeDescriptionFormatter(), 5000),
            (f'{stem}_video_analysis_report.md',          MarkdownReportFormatter(),     None),
            (f'{stem}_video_analysis.json',               JsonFormatter(),               None),
            (f'{stem}_video_summary.txt',                 SimpleTextFormatter(),         None),
        ]
        for filename, formatter, max_len in analysis_outputs:
            output_file = output_folder / filename
            kwargs = dict(analysis=analysis, project=project)
            if max_len is not None:
                kwargs['max_length'] = max_len
            content = formatter.format(**kwargs)
            output_file.write_text(content, encoding='utf-8')
            logger.success(f"Generated {filename}")
            generated.append(filename)

        # Transcript-based outputs
        transcript_outputs = [
            (f'{stem}_transcript.txt',               PlainTextTranscriptFormatter()),
            (f'{stem}_transcript.srt',               SrtTranscriptFormatter()),
            (f'{stem}_transcript_w_timestamps.md',   MarkdownTranscriptFormatter()),
        ]
        for filename, formatter in transcript_outputs:
            output_file = output_folder / filename
            content = formatter.format(transcripts=transcripts, project=project)
            output_file.write_text(content, encoding='utf-8')
            logger.success(f"Generated {filename}")
            generated.append(filename)

        return generated


class PipelineResult(BaseModel):
    """Result of pipeline processing."""
    project: VideoProject
    stats: ProcessingStats
    analysis: Optional[FullVideoAnalysis] = None

    def summary_report(self) -> str:
        """Generate a summary report."""
        return f"""
Pipeline Results for {self.project.title or self.project.video_path.name}

{'=' * 60}

Processing Statistics:
- Audio chunks: {self.stats.audio_chunks_created} created, {self.stats.audio_chunks_cached} cached
- Transcripts: {self.stats.transcripts_created} created, {self.stats.transcripts_cached} cached  
- Analyses: {self.stats.analyses_created} created, {self.stats.analyses_cached} cached
- Cache hit rate: {self.stats.cache_hit_rate:.1%}
- Total duration: {self.stats.total_duration_seconds:.1f}s

{'=' * 60}
Find the results at: \n\n{self.project.output_folder}\n\n

"""