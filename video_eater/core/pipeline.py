# pipeline.py
from typing import Optional

import yaml
from pydantic import BaseModel

from video_eater.core.handle_video.audio_extractor import AudioExtractor
from video_eater.core.config_models import VideoProject, ProcessingStats, ProcessingConfig, TranscriptionProvider
from video_eater.core.output_templates import YouTubeDescriptionFormatter, MarkdownReportFormatter, JsonFormatter, \
    SimpleTextFormatter
from video_eater.core.transcribe_audio.transcribe_audio_chunks import transcribe_audio_chunk_folder
from video_eater.core.ai_processors.transcript_processor import TranscriptProcessor
from video_eater.core.ai_processors.ai_prompt_models import FullVideoAnalysis
from video_eater.logging_config import PipelineLogger


class VideoProcessingPipeline:
    """Clean, maintainable video processing pipeline."""

    def __init__(self,
                 config: ProcessingConfig):
        self.config = config
        self.pipeline_logger = PipelineLogger()
        self.stats = ProcessingStats()

    async def process_video(self, project: VideoProject):
        """Process a single video through the pipeline."""

        self.pipeline_logger.step(1, 3, f"Processing: {project.video_path.name}")

        # Each step is now clean and focused
        audio_chunks = await self._extract_audio(project)
        transcripts = await self._transcribe_chunks(project)
        analysis = await self._analyze_transcripts(project, transcripts)
        _ = await self._generate_outputs(project, analysis)
        return PipelineResult(
            project=project,
            stats=self.stats,
            analysis=analysis
        )
    
    async def _extract_audio(self, project: VideoProject):
        """Extract audio with clean separation of concerns."""

        extractor = AudioExtractor(
            chunk_length=self.config.chunk_length_seconds,
            overlap=self.config.chunk_overlap_seconds
        )

        # Check cache
        if not self.config.force_chunk_audio:
            existing = extractor.find_existing_chunks(project.audio_chunks_folder)
            if existing:
                self.pipeline_logger.cache_hit(f"{len(existing)} audio chunks")
                self.stats.audio_chunks_cached = len(existing)
                return existing

        # Extract new chunks
        chunks = await extractor.extract(
            video_path=project.video_path,
            output_folder=project.audio_chunks_folder
        )

        self.stats.audio_chunks_created = len(chunks)
        self.pipeline_logger.success(f"Created {len(chunks)} audio chunks")

        return chunks

    async def _transcribe_chunks(self, project: VideoProject):
        """Transcribe audio chunks and return transcripts."""
        project.transcript_chunks_folder.mkdir(parents=True, exist_ok=True)

        before_count = len(list(project.transcript_chunks_folder.glob("*.transcript.json")))

        use_local = self.config.transcription_provider == TranscriptionProvider.LOCAL_WHISPER
        use_aai = self.config.transcription_provider == TranscriptionProvider.ASSEMBLY_AI

        transcripts = await transcribe_audio_chunk_folder(
            audio_chunk_folder=str(project.audio_chunks_folder),
            transcript_chunk_folder=str(project.transcript_chunks_folder),
            file_extension=".mp3",
            re_transcribe=self.config.force_transcribe,
            local_whisper=use_local,
            use_assembly_ai=use_aai,
        )

        after_count = len(list(project.transcript_chunks_folder.glob("*.transcript.json")))
        created = max(0, after_count - before_count) if not self.config.force_transcribe else after_count
        cached = 0 if self.config.force_transcribe else before_count
        self.stats.transcripts_created = created
        self.stats.transcripts_cached = cached
        self.pipeline_logger.success(f"Prepared {after_count} transcript chunks ({created} new, {cached} cached)")
        return transcripts

    async def _analyze_transcripts(self, project: VideoProject, transcripts):
        """Analyze transcripts into a full video analysis."""
        processor = TranscriptProcessor(
            model=self.config.analysis_model,
            use_async=True,
            max_concurrent_chunks=self.config.max_concurrent_chunks,
            batch_size=self.config.batch_size,
            chunk_length_seconds=self.config.chunk_length_seconds,
            chunk_overlap_seconds=self.config.chunk_overlap_seconds,
        )

        chunk_analyses = await processor.process_transcript_folder(
            transcript_folder=project.transcript_chunks_folder,
            chunk_analysis_output_folder=project.analysis_folder,
        )
        # Combine all analyses (or load from cache if already done)
        combined_file = project.video_path.parent / "full_video_analysis.yaml"


        if combined_file.exists():
            print(f"\n📂 Using cached full video analysis from {combined_file}")
            with open(combined_file, 'r', encoding='utf-8') as f:
                full_analysis = FullVideoAnalysis(**yaml.safe_load(f))
            return full_analysis

        print("\n🔄 Combining all chunk analyses...")
        full_analysis = await processor.combine_analyses(chunk_analyses)

        # Save combined analysis as YAML
        with open(combined_file, 'w', encoding='utf-8') as f:
            yaml.dump(full_analysis.model_dump(), f,
                      default_flow_style=False,
                      sort_keys=False,
                      allow_unicode=True)
        print(f"💾 Saved combined analysis to {combined_file}")


        return full_analysis


    async def _generate_outputs(self, project: VideoProject, analysis: FullVideoAnalysis):
        """Generate outputs using configurable formatters."""

        # Define which formatters to use (this could come from config)
        formatters = {
            f'{project.video_path.stem}_youtube_description.txt': YouTubeDescriptionFormatter(),
            f'{project.video_path.stem}_video_analysis_report.md': MarkdownReportFormatter(),
            f'{project.video_path.stem}_video_analysis.json': JsonFormatter(),
            f'{project.video_path.stem}_video_summary.txt': SimpleTextFormatter(),
        }


        output_folder = project.output_folder
        output_folder.mkdir(parents=True, exist_ok=True)

        for filename, formatter in formatters.items():
            output_file = output_folder / filename
            content = formatter.format(analysis)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            self.pipeline_logger.success(f"Generated {filename}")

        return list(formatters.keys())



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

"""
