from video_eater.core.ai_processors.base_processor import BaseAIProcessor
import asyncio
from pathlib import Path
from typing import List, Tuple
from pathlib import Path
from openai.types.audio import TranscriptionVerbose

from video_eater.core.transcribe_audio.transcript_models import VideoTranscript


async def transcribe_audio_chunk_folder(chunk_folder: str, file_extension: str = ".mp3", re_transcribe:bool=True) -> list[VideoTranscript]:
    chunk_paths = list(Path(chunk_folder).glob(f"*{file_extension}"))

    paths_to_process = []
    if not re_transcribe:
        for path in chunk_paths:
            if path.is_file():
                continue
            paths_to_process.append(path)
    else:
        paths_to_process = chunk_paths

    return await transcribe_audio_chunks([str(path) for path in paths_to_process])

async def _transcribe_single_chunk(chunk_path: str, transcript_output_json_path:Path) -> tuple[TranscriptionVerbose, str]:
    ai = BaseAIProcessor(use_async=True)  
    transcript = await ai.async_make_whisper_transcription_request(audio_file_path=chunk_path)
    transcript_output_json_path.write_text(transcript.model_dump_json(indent=2))
    print(f"Saved transcript for {str(transcript_output_json_path)}")
    
    # Return both the transcript text and the path to the saved JSON
    return transcript, str(transcript_output_json_path)

async def transcribe_audio_chunks(chunk_paths: List[str], reprocess_all:bool=False) -> list[VideoTranscript]:
    print(f"Transcribing {len(chunk_paths)} audio chunks...")
    
    sub_transcript_tasks = []

    # Create transcript path and save the JSON
    chunk_folder = Path(chunk_paths[0]).parent.parent
    transcript_folder = chunk_folder/ 'chunk_transcripts'
    transcript_folder.mkdir(parents=True, exist_ok=True)

    for chunk_path in chunk_paths:
        transcript_json_filename = Path(chunk_path).name.replace('.mp3', '.transcript.json')
        transcript_path = transcript_folder / transcript_json_filename
        if transcript_path.exists() and not reprocess_all:
            print(f"Transcript already exists for {chunk_path}, skipping...")
            continue
        
        sub_transcript_tasks.append(asyncio.create_task(_transcribe_single_chunk(chunk_path=chunk_path,
                                                                transcript_output_json_path=transcript_path)))
    
    # Gather results from all tasks
    results = await asyncio.gather(*sub_transcript_tasks)
    
    return [VideoTranscript.from_openai_transcript(result[0]) for result in results]

