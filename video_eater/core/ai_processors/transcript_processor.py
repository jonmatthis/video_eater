"""Process transcribed chunks to generate summaries, outlines, and chapters.

Thin facade composing ChunkAnalyzer, ChunkBatchProcessor, and AnalysisSynthesizer.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import yaml

from video_eater.core.ai_processors.ai_prompt_models import (
    FullVideoAnalysis, ChunkAnalysisWithTimestamp, StartingTimeString,
)
from video_eater.core.ai_processors.base_processor import BaseAIProcessor
from video_eater.core.ai_processors.chunk_analyzer import ChunkAnalyzer
from video_eater.core.ai_processors.analysis_synthesizer import AnalysisSynthesizer
from video_eater.core.ai_processors.transcript_utils import (
    format_timestamp, parse_chunk_filename, _return_cached,
)
from video_eater.core.transcribe_audio.transcript_models import VideoTranscript

logger = logging.getLogger(__name__)


class TranscriptProcessor:
    """Process transcribed chunks to generate summaries, outlines, and chapters.

    This is a facade that composes three focused classes:
      - ChunkAnalyzer: single-chunk LLM inference
      - ChunkBatchProcessor (private): batching, caching, concurrency
      - AnalysisSynthesizer: cross-chunk synthesis
    """

    def __init__(self,
                 model: str,
                 use_async: bool = True,
                 max_concurrent_chunks: int = 50,
                 batch_size: int = 10,
                 chunk_length_seconds: int = 600):
        self._llm = BaseAIProcessor(model=model, use_async=use_async)
        self._chunk_analyzer = ChunkAnalyzer(processor=self._llm, chunk_length_seconds=chunk_length_seconds)
        self._synthesizer = AnalysisSynthesizer(processor=self._llm)

        self._max_concurrent_chunks = max_concurrent_chunks
        self._batch_size = batch_size
        self._chunk_length_seconds = chunk_length_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent_chunks)
        self.processing_stats = {
            'chunks_processed': 0,
            'chunks_cached': 0,
            'processing_time': 0,
            'errors': []
        }

    # ------------------------------------------------------------------
    # Public API (matches old TranscriptProcessor for pipeline compat)
    # ------------------------------------------------------------------

    async def process_transcript_folder(
        self,
        transcript_folder: Path,
        chunk_analysis_output_folder: Path,
    ) -> list[ChunkAnalysisWithTimestamp]:
        """Process all transcript chunks in a folder with parallel processing."""
        return await self._process_transcript_folder(transcript_folder, chunk_analysis_output_folder)

    async def combine_analyses(
        self,
        chunk_analyses: list[ChunkAnalysisWithTimestamp],
    ) -> FullVideoAnalysis:
        """Combine individual chunk analyses into a complete video analysis."""
        return await self._synthesizer.combine_analyses(chunk_analyses)

    async def close(self):
        """Close the underlying LLM client connection pool."""
        await self._llm.close()

    # ------------------------------------------------------------------
    # Private: batching, caching, concurrency orchestration
    # ------------------------------------------------------------------

    async def _analyze_with_semaphore(
        self,
        transcript_data: VideoTranscript,
        chunk_start_time_string: StartingTimeString,
        chunk_index: int,
        chunk_name: str,
    ) -> Tuple[int, ChunkAnalysisWithTimestamp]:
        """Rate-limited wrapper around ChunkAnalyzer.analyze_chunk."""
        async with self._semaphore:
            try:
                t0 = time.time()
                logger.info(
                    f"Processing chunk {chunk_index + 1}: {chunk_name} "
                    f"(starts at {format_timestamp(transcript_data.start_time)})")

                analysis = await self._chunk_analyzer.analyze_chunk(
                    transcript_data=transcript_data,
                    chunk_start_time_string=chunk_start_time_string,
                    chunk_index=chunk_index,
                )

                elapsed = time.time() - t0
                logger.success(f"Completed chunk {chunk_index + 1} in {elapsed:.1f}s")
                return chunk_index, analysis

            except Exception as e:
                error_msg = f"Error analyzing chunk {chunk_index}: {str(e)}"
                logger.error(f"Failed chunk {chunk_index + 1}: {e}")
                self.processing_stats['errors'].append(error_msg)
                raise

    async def _process_transcript_batch(
        self,
        batch: list[tuple[int, StartingTimeString, Path]],
        output_folder: Path,
    ) -> List[ChunkAnalysisWithTimestamp]:
        """Process a batch of transcript files in parallel."""
        tasks = []

        for chunk_idx, chunk_start_time_string, transcript_file in batch:
            analysis_file = output_folder / f"{transcript_file.stem}.analysis.yaml"

            if analysis_file.exists():
                logger.debug(f"Using cached analysis for chunk {chunk_idx + 1}")
                self.processing_stats['chunks_cached'] += 1
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    chunk_analysis = ChunkAnalysisWithTimestamp(**yaml.safe_load(f))
                tasks.append(asyncio.create_task(
                    _return_cached(chunk_idx, chunk_analysis)
                ))
            else:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript_data = VideoTranscript(**json.load(f))

                task = asyncio.create_task(
                    self._analyze_with_semaphore(
                        transcript_data=transcript_data,
                        chunk_start_time_string=chunk_start_time_string,
                        chunk_index=chunk_idx,
                        chunk_name=transcript_file.name,
                    )
                )
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        chunk_analyses: list[Tuple[int, ChunkAnalysisWithTimestamp]] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch task failed: {result}")
                self.processing_stats['errors'].append(str(result))
                continue

            idx, analysis = result
            chunk_analyses.append((idx, analysis))

            batch_item = next((item for item in batch if item[0] == idx), None)
            if batch_item:
                _, _, transcript_file = batch_item
                analysis_file = output_folder / f"{transcript_file.stem}.analysis.yaml"
                if not analysis_file.exists():
                    with open(analysis_file, 'w', encoding='utf-8') as f:
                        yaml.dump(analysis.model_dump(), f,
                                  default_flow_style=False,
                                  sort_keys=False,
                                  allow_unicode=True)
                    self.processing_stats['chunks_processed'] += 1

        chunk_analyses.sort(key=lambda x: x[0])
        return [analysis for _, analysis in chunk_analyses]

    async def _process_transcript_folder(
        self,
        transcript_folder: Path,
        chunk_analysis_output_folder: Path,
    ) -> list[ChunkAnalysisWithTimestamp]:
        """Discover transcript files, batch them, and process all."""
        t0 = time.perf_counter()

        chunk_analysis_output_folder.mkdir(parents=True, exist_ok=True)

        transcript_files = sorted(transcript_folder.glob("*.transcript.json"))
        if not transcript_files:
            raise ValueError(f"No transcript files found in {transcript_folder}")

        logger.info(
            f"Processing {len(transcript_files)} transcript files "
            f"(max concurrent: {self._max_concurrent_chunks}, batch: {self._batch_size}, "
            f"chunk: {self._chunk_length_seconds}s)")

        file_data: list[tuple[int, StartingTimeString, Path]] = []
        for transcript_file in transcript_files:
            chunk_index, _, start_time_string = parse_chunk_filename(transcript_file.name)
            file_data.append((chunk_index, start_time_string, transcript_file))

        all_chunk_analyses: list[ChunkAnalysisWithTimestamp] = []
        for batch_idx in range(0, len(file_data), self._batch_size):
            batch = file_data[batch_idx:batch_idx + self._batch_size]
            batch_num = batch_idx // self._batch_size + 1
            batch_analyses = await self._process_transcript_batch(batch, chunk_analysis_output_folder)
            all_chunk_analyses.extend(batch_analyses)
            logger.success(f"Batch {batch_num} complete")

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Processing Statistics: {elapsed:.1f}s total, "
            f"{self.processing_stats['chunks_processed']} processed, "
            f"{self.processing_stats['chunks_cached']} cached, "
            f"{elapsed / len(transcript_files):.1f}s avg")

        if self.processing_stats['errors']:
            logger.warning(f"Errors: {len(self.processing_stats['errors'])}")
            for error in self.processing_stats['errors'][:5]:
                logger.warning(f"  {error}")

        return all_chunk_analyses
