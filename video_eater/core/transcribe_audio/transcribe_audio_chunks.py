import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Tuple

from openai import AsyncOpenAI

from video_eater.core.transcribe_audio.transcript_models import VideoTranscript

logger = logging.getLogger(__name__)


def _parse_chunk_offset(filename: str) -> float:
    """Extract chunk start time from filename like '*_chunk_000__123.45sec*'."""
    match = re.search(r"chunk_\d{3}__(\d+(?:\.\d+)?)sec", filename)
    if match:
        return float(match.group(1))
    return 0.0


async def transcribe_audio_chunk_folder(
        audio_chunk_folder: str,
        transcript_chunk_folder: str,
        file_extension: str = ".mp3",
        re_transcribe: bool = False,
        whisper_prompt: str | None = None,
) -> List[VideoTranscript]:
    """
    Transcribe all audio chunks in a folder using Groq whisper.

    Args:
        audio_chunk_folder: Path to folder containing audio chunks
        transcript_chunk_folder: Path to folder for transcript output
        file_extension: File extension of audio files
        re_transcribe: If True, re-transcribe even if transcripts exist

    Returns:
        List of VideoTranscript objects
    """
    chunk_folder_path = Path(audio_chunk_folder)
    transcript_folder = Path(transcript_chunk_folder)
    transcript_folder.mkdir(parents=True, exist_ok=True)

    # Find all audio chunks
    audio_chunks = sorted(list(chunk_folder_path.glob(f"*{file_extension}")))

    if not audio_chunks:
        logger.warning(f"No audio files found in {audio_chunk_folder}")
        return []

    # Determine which chunks need transcription
    chunks_to_transcribe = []
    existing_transcripts = []

    for chunk_path in audio_chunks:
        transcript_filename = chunk_path.name.replace('.mp3', '.transcript.json')
        transcript_path = transcript_folder / transcript_filename

        if transcript_path.exists() and not re_transcribe:
            logger.debug(f"Transcript exists: {chunk_path.name}")
            existing_transcripts.append(transcript_path)
        else:
            chunks_to_transcribe.append((chunk_path, transcript_path))

    logger.info(f"Transcription Summary: {len(audio_chunks)} total, {len(existing_transcripts)} cached, {len(chunks_to_transcribe)} to transcribe")

    # Transcribe missing chunks
    if chunks_to_transcribe:
        logger.info(f"Transcribing {len(chunks_to_transcribe)} audio chunks...")
        client = AsyncOpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            timeout=600,
        )
        try:
            new_transcripts = await transcribe_audio_chunks(
                chunk_paths=chunks_to_transcribe,
                reprocess_all=re_transcribe,
                client=client,
                whisper_prompt=whisper_prompt,
            )
        finally:
            await client.close()
    else:
        logger.info("All chunks already transcribed")
        new_transcripts = []

    # Load all transcripts (existing + new)
    all_transcripts = []

    # Load existing transcripts
    for transcript_path in existing_transcripts:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
            try:
                transcript = VideoTranscript(**transcript_data)
                transcript.chunk_offset_seconds = _parse_chunk_offset(transcript_path.name)
                all_transcripts.append(transcript)
            except Exception as e:
                logger.warning(f"Failed to load cached transcript {transcript_path.name}: {e}")

    # Add new transcripts
    for transcript in new_transcripts:
        # chunk_offset_seconds is set during _transcribe_single_chunk
        all_transcripts.append(transcript)

    return all_transcripts


async def _transcribe_single_chunk(
        chunk_path: Path,
        transcript_output_json_path: Path,
        chunk_index: int,
        total_chunks: int,
        client: AsyncOpenAI,
        whisper_prompt: str | None = None) -> Tuple[VideoTranscript, str]:
    """
    Transcribe a single audio chunk using Groq whisper.

    Args:
        chunk_path: Path to audio chunk
        transcript_output_json_path: Path to save transcript JSON
        chunk_index: Index of current chunk (for progress display)
        total_chunks: Total number of chunks (for progress display)
        client: Shared AsyncOpenAI client (Groq)
        whisper_prompt: Optional prompt to guide vocabulary/transcription

    Returns:
        Tuple of (transcript, path_to_saved_json)
    """
    logger.debug(f"[{chunk_index}/{total_chunks}] Transcribing: {chunk_path.name}")

    with open(chunk_path, "rb") as audio_file:
        kwargs = dict(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
        if whisper_prompt:
            kwargs["prompt"] = whisper_prompt
        transcript_response = await client.audio.transcriptions.create(**kwargs)

    transcript = VideoTranscript.from_whisper_response(transcript_response)
    transcript.chunk_offset_seconds = _parse_chunk_offset(chunk_path.name)

    # Save transcript
    transcript_output_json_path.write_text(
        transcript.model_dump_json(indent=2),
        encoding='utf-8'
    )

    print(f"  [{chunk_index}/{total_chunks}] ✓ Saved: {transcript_output_json_path.name}")

    return transcript, str(transcript_output_json_path)


async def transcribe_audio_chunks(
        chunk_paths: List[Tuple[Path, Path]],
        reprocess_all: bool = False,
        max_concurrent: int = 50,
        client: AsyncOpenAI | None = None,
        whisper_prompt: str | None = None) -> List[VideoTranscript]:
    """
    Transcribe multiple audio chunks with concurrency control.

    Args:
        chunk_paths: List of tuples (audio_path, transcript_path)
        reprocess_all: If True, reprocess even existing transcripts
        max_concurrent: Maximum number of concurrent transcriptions

    Returns:
        List of VideoTranscript objects for newly transcribed chunks
    """
    if not chunk_paths:
        return []

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    async def transcribe_with_semaphore(
            audio_path: Path,
            transcript_path: Path,
            index: int,
    ) -> Tuple[VideoTranscript, str]:
        async with semaphore:
            return await _transcribe_single_chunk(
                chunk_path=audio_path,
                transcript_output_json_path=transcript_path,
                chunk_index=index,
                total_chunks=len(chunk_paths),
                client=client,
                whisper_prompt=whisper_prompt,
            )

    # Create tasks for all chunks
    tasks = []
    for chunk_number, (audio_path, transcript_path) in enumerate(chunk_paths, 1):
        task = asyncio.create_task(
            transcribe_with_semaphore(audio_path=audio_path,
                                      transcript_path=transcript_path,
                                      index=chunk_number,
                                      )
        )
        tasks.append(task)

    # Wait for all tasks to complete
    logger.info(f"Processing with max {max_concurrent} concurrent transcriptions...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    transcripts = []
    errors = []

    for chunk_number, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append((chunk_paths[chunk_number][0].name, str(result)))
        else:
            transcript, _ = result
            transcripts.append(transcript)

    # Report any errors
    if errors:
        for filename, error in errors:
            logger.error(f"Transcription error: {filename}: {error}")
        raise RuntimeError(f"Transcription errors occurred: {len(errors)} chunks failed")

    logger.success(f"Successfully transcribed {len(transcripts)} chunks")

    return transcripts


def get_transcription_status(chunk_folder: str, file_extension: str = ".mp3") -> dict:
    """
    Get the transcription status for a folder of audio chunks.

    Args:
        chunk_folder: Path to folder containing audio chunks
        file_extension: File extension of audio files

    Returns:
        Dictionary with status information
    """
    chunk_folder_path = Path(chunk_folder)
    transcript_folder = chunk_folder_path.parent / 'chunk_transcripts'

    audio_chunks = list(chunk_folder_path.glob(f"*{file_extension}"))

    transcribed = []
    not_transcribed = []

    for chunk_path in audio_chunks:
        transcript_filename = chunk_path.name.replace('.mp3', '.transcript.json')
        transcript_path = transcript_folder / transcript_filename

        if transcript_path.exists():
            transcribed.append(chunk_path.name)
        else:
            not_transcribed.append(chunk_path.name)

    return {
        'total_chunks': len(audio_chunks),
        'transcribed': len(transcribed),
        'not_transcribed': len(not_transcribed),
        'transcribed_files': transcribed,
        'not_transcribed_files': not_transcribed,
        'all_transcribed': len(not_transcribed) == 0
    }
