"""Module-level utilities extracted from TranscriptProcessor."""

import asyncio
import re
from typing import Tuple

from video_eater.core.ai_processors.ai_prompt_models import StartingTimeString, ChunkAnalysisWithTimestamp


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    from video_eater.core.output_templates import format_timestamp_hhmmss
    return format_timestamp_hhmmss(seconds)


def parse_chunk_filename(filename: str) -> Tuple[int, float, StartingTimeString]:
    """Extract (chunk_index, start_time_seconds, start_time_string) from a chunk filename.

    Expected format: *_chunk_XXX__YYY.Ysec* where XXX is the 0-padded index
    and YYY.Y is the chunk start time in seconds.

    Raises ValueError if the filename does not match the expected pattern.
    """
    pattern = r"chunk_(\d{3})__(\d+(?:\.\d+)?)sec"
    match = re.search(pattern, filename)
    if match:
        chunk_index = int(match.group(1))
        start_time_string = match.group(2)
        start_time_float = float(start_time_string)
        return chunk_index, start_time_float, start_time_string

    raise ValueError(f"Filename does not match expected pattern: {filename}")


async def _return_cached(idx: int, analysis: ChunkAnalysisWithTimestamp) -> Tuple[int, ChunkAnalysisWithTimestamp]:
    """Helper to return cached analysis in async context."""
    return idx, analysis
