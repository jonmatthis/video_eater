from video_eater.core.ai_processors.base_processor import BaseAIProcessor
import asyncio
from pathlib import Path
from typing import List, Tuple
from pathlib import Path
from openai.types.audio import TranscriptionVerbose

async def transcribe_audio_chunk_folder(chunk_folder: str, file_extension: str = ".mp3") -> Tuple[str, TranscriptionVerbose, List[str]]:
    chunk_paths = list(Path(chunk_folder).glob(f"*{file_extension}"))
    return await transcribe_audio_chunks([str(path) for path in chunk_paths])

async def _transcribe_single_chunk(chunk_path: str) -> tuple[str, str]:
    ai = BaseAIProcessor(use_async=True)  
    transcript = await ai.async_make_whisper_transcription_request(audio_file_path=chunk_path)
    
    # Create transcript path and save the JSON
    chunk_folder = Path(chunk_path).parent
    transcript_folder = chunk_folder/ 'chunk_transcripts'
    transcript_folder.mkdir(parents=True, exist_ok=True)
    transcript_json_filename = Path(chunk_path).name.replace('.mp3', '.transcript.json')
    transcript_path = transcript_folder / transcript_json_filename
    transcript_path.write_text(transcript.model_dump_json(indent=2))
    
    # Return both the transcript text and the path to the saved JSON
    return transcript

async def transcribe_audio_chunks(chunk_paths: List[str]) -> list[TranscriptionVerbose]:
    print(f"Transcribing {len(chunk_paths)} audio chunks...")
    
    sub_transcript_tasks = []
    
    for chunk_path in chunk_paths:
        sub_transcript_tasks.append(_transcribe_single_chunk(chunk_path))
    
    # Gather results from all tasks
    results = await asyncio.gather(*sub_transcript_tasks)
    
    return results

