import asyncio
import json
import os
from pathlib import Path
from typing import List, Tuple
from xml.dom import ValidationErr

from openai.types.audio import TranscriptionVerbose

from video_eater.core.ai_processors.base_processor import BaseAIProcessor
from video_eater.core.transcribe_audio.local_whisper_audio_transcription import transcribe_audio_with_local_whisper
from video_eater.core.transcribe_audio.transcribe_audio_assembly_ai import async_transcribe_with_assemblyai
from video_eater.core.transcribe_audio.transcript_models import VideoTranscript


async def transcribe_audio_chunk_folder(
        audio_chunk_folder: str,
        transcript_chunk_folder: str,
        file_extension: str = ".mp3",
        re_transcribe: bool = False,
        local_whisper: bool = False,
        use_assembly_ai: bool = True
) -> List[VideoTranscript]:
    """
    Transcribe all audio chunks in a folder.

    Args:
        audio_chunk_folder: Path to folder containing audio chunks
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
        print(f"⚠️ No audio files found in {audio_chunk_folder}")
        return []

    # Determine which chunks need transcription
    chunks_to_transcribe = []
    existing_transcripts = []

    for chunk_path in audio_chunks:
        transcript_filename = chunk_path.name.replace('.mp3', '.transcript.json')
        transcript_path = transcript_folder / transcript_filename

        if transcript_path.exists() and not re_transcribe:
            print(f"  ✓ Transcript exists: {chunk_path.name}")
            existing_transcripts.append(transcript_path)
        else:
            chunks_to_transcribe.append((chunk_path, transcript_path))

    print(f"\n📊 Transcription Summary:")
    print(f"   • Total audio chunks: {len(audio_chunks)}")
    print(f"   • Existing transcripts: {len(existing_transcripts)}")
    print(f"   • Need transcription: {len(chunks_to_transcribe)}")

    # Transcribe missing chunks
    if chunks_to_transcribe:
        print(f"\n🎙️ Transcribing {len(chunks_to_transcribe)} audio chunks...")
        new_transcripts = await transcribe_audio_chunks(
            chunk_paths=chunks_to_transcribe,
            reprocess_all=re_transcribe,
            local_whisper=local_whisper,
            use_assembly_ai=True
        )
    else:
        print(f"   ℹ️ All chunks already transcribed")
        new_transcripts = []

    # Load all transcripts (existing + new)
    all_transcripts = []

    # Load existing transcripts
    for transcript_path in existing_transcripts:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
            try:
                all_transcripts.append(VideoTranscript.from_openai_transcript(
                    TranscriptionVerbose(**transcript_data)
                ))
            except Exception as e:
                all_transcripts.append(VideoTranscript(**transcript_data))


    # Add new transcripts
    all_transcripts.extend(new_transcripts)

    return all_transcripts


async def _transcribe_single_chunk(
        chunk_path: Path,
        transcript_output_json_path: Path,
        chunk_index: int,
        total_chunks: int,
        local_whisper: bool,
        use_assembly_ai: bool = True) -> Tuple[TranscriptionVerbose|VideoTranscript, str]:
    """
    Transcribe a single audio chunk.

    Args:
        chunk_path: Path to audio chunk
        transcript_output_json_path: Path to save transcript JSON
        chunk_index: Index of current chunk (for progress display)
        total_chunks: Total number of chunks (for progress display)

    Returns:
        Tuple of (transcript, path_to_saved_json)
    """
    print(f"  [{chunk_index}/{total_chunks}] Transcribing: {chunk_path.name}")
    if local_whisper:
        transcript = transcribe_audio_with_local_whisper(audio_path=str(chunk_path),
                                                         model_name="large")
    elif use_assembly_ai:
        transcript = await async_transcribe_with_assemblyai(str(chunk_path), os.getenv("ASSEMBLY_AI_API_KEY"))
    else:
        ai = BaseAIProcessor(use_async=True)
        transcript = await ai.async_make_whisper_transcription_request(
            audio_file_path=str(chunk_path)
        )

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
        local_whisper: bool = False,
        use_assembly_ai: bool = True,
        max_concurrent: int = 50) -> List[VideoTranscript]:
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
            index: int
    ) -> Tuple[TranscriptionVerbose|VideoTranscript, str]:
        async with semaphore:
            return await _transcribe_single_chunk(
                chunk_path=audio_path,
                transcript_output_json_path=transcript_path,
                local_whisper=local_whisper,
                use_assembly_ai=use_assembly_ai,
                chunk_index=index,
                total_chunks=len(chunk_paths)
            )

    # Create tasks for all chunks
    tasks = []
    for i, (audio_path, transcript_path) in enumerate(chunk_paths, 1):

        task = asyncio.create_task(
            transcribe_with_semaphore(audio_path, transcript_path, i)
        )
        tasks.append(task)

    # Wait for all tasks to complete
    print(f"\n⚡ Processing with max {max_concurrent} concurrent transcriptions...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    transcripts = []
    errors = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append((chunk_paths[i][0].name, str(result)))
        else:
            transcript, _ = result
            if isinstance(transcript, TranscriptionVerbose):
                transcripts.append(VideoTranscript.from_openai_transcript(transcript))
            elif isinstance(transcript,VideoTranscript):
                transcripts.append(transcript)

    # Report any errors
    if errors:
        print(f"\n⚠️ Transcription errors occurred:")
        for filename, error in errors:
            print(f"   • {filename}: {error}")
        raise RuntimeError(f"\n⚠️ Transcription errors occurred:")

    print(f"\n✅ Successfully transcribed {len(transcripts)} chunks")

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
