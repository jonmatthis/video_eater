import asyncio
import logging
import yaml
from pathlib import Path
from typing import List, Tuple

from .base_processor import BaseProcessor
from ..cache_stuff import CACHE_DIRS
from ..yt_models import VideoTranscript, ProcessedTranscript, TranscriptEntry
from ..yt_prompts import CLEANUP_TRANSCRIPT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

class YoutubeTranscriptCleaner(BaseProcessor):
    """Responsible for cleaning raw transcripts."""

    async def process_all_transcripts(self) -> List[ProcessedTranscript]:
        """Clean all raw transcripts from the cache directory."""
        logger.info("Starting transcript cleaning")
        raw_files = list(CACHE_DIRS['raw'].glob('*.yaml'))
        processed_transcripts = []
        tasks = []

        # First load any existing cleaned transcripts that don't need refreshing
        for raw_file in raw_files:
            output_file = CACHE_DIRS['cleaned'] / raw_file.name
            if not self.force_refresh and output_file.exists():
                logger.debug(f"Loading existing cleaned transcript: {raw_file.stem}")
                processed = ProcessedTranscript(**yaml.safe_load(output_file.read_text()))
                processed_transcripts.append(processed)
                continue

            # Create task for files that need processing
            video_data = VideoTranscript(**yaml.safe_load(raw_file.read_text()))
            tasks.append(self._process_transcript_file(
                video_data=video_data,
                output_file=output_file
            ))

        # Process all remaining transcripts in parallel
        if tasks:
            CACHE_DIRS['cleaned'].mkdir(exist_ok=True, parents=True)
            results = await asyncio.gather(*tasks)
            processed_transcripts.extend(results)

        return processed_transcripts

    async def _process_transcript_file(self,
                                      video_data: VideoTranscript,
                                      output_file: Path) -> ProcessedTranscript:
        """Process a single transcript file and save the result."""
        processed = await self.process_transcript(video_data)

        # Save the processed transcript
        with open(output_file, 'w') as f:
            yaml.dump(processed.model_dump(), f)

        return processed

    async def process_transcript(self, video_data: VideoTranscript) -> ProcessedTranscript:
        """Clean a single video transcript using AI."""
        logger.info(f"Cleaning transcript for {video_data.metadata.title}")

        tasks = [
            self._clean_chunk(i, chunk, video_data.metadata.title, len(video_data.transcript_chunks))
            for i, chunk in enumerate(video_data.transcript_chunks)
        ]
        chunks = await asyncio.gather(*tasks)

        return ProcessedTranscript(
            video_id=video_data.video_id,
            title=video_data.metadata.title,
            transcript_chunks=chunks,
            full_transcript=" ".join([c.text for c in chunks])
        )

    async def _clean_chunk(self,
                           chunk_number: int,
                           chunk: TranscriptEntry,
                           title: str,
                           total_chunks: int) -> TranscriptEntry:
        """Clean an individual transcript chunk."""
        logger.debug(
            f"Cleaning chunk {chunk_number + 1}/{total_chunks} for {title}")
        cleaned_chunk = await self.make_openai_json_mode_ai_request(
            system_prompt=CLEANUP_TRANSCRIPT_SYSTEM_PROMPT,
            input_data=chunk.model_dump(),
            output_model=TranscriptEntry
        )
        return cleaned_chunk