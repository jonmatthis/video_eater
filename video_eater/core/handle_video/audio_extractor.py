# audio_extractor.py
from __future__ import annotations
import asyncio
from pathlib import Path
from typing import List

from video_eater.core.transcribe_audio.extract_audio_chunks import (
    extract_audio_from_video,
)


class AudioExtractor:
    """Adapter around existing ffmpeg-based helpers to fit the new pipeline API."""

    def __init__(self, chunk_length: int, overlap: int):
        self.chunk_length = chunk_length
        self.overlap = overlap

    def find_existing_chunks(self, chunks_folder: Path) -> List[Path]:
        """Return existing chunk files in the expected folder."""
        if not chunks_folder.exists():
            return []
        # Match the naming used by chunk_audio_file: *_chunk_XXX_...mp3
        return sorted(chunks_folder.glob("*_chunk_*.mp3"))

    async def extract(self, video_path: Path, output_folder: Path) -> List[Path]:
        """Extract and chunk audio asynchronously (delegates to blocking code in a thread)."""
        output_folder.mkdir(parents=True, exist_ok=True)

        def _run_blocking() -> List[Path]:
            _, chunk_files = extract_audio_from_video(
                video_file=str(video_path),
                audio_output_file=None,  # default alongside input name
                chunk_audio=True,
                audio_chunk_folder=str(output_folder),
                chunk_length_seconds=float(self.chunk_length),
                chunk_overlap_seconds=float(self.overlap),
            )
            return [Path(p) for p in chunk_files]

        return await asyncio.to_thread(_run_blocking)

