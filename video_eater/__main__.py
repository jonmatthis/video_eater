import sys 
import os 
import asyncio
from pathlib import Path
# Add the parent directory to sys.path to make the package importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from video_eater.core.transcribe_audio.transcribe_audio_chunks import transcribe_audio_chunk_folder
from video_eater.core.transcribe_audio.extract_audio_chunks import extract_audio_from_video


VIDEO_PATH  = r"\\jon-nas\jon-nas\videos\livestream_videos\2025-08-07-JSM-Livestream\2025-08-07-JSM-Livestream-RAW.mp4"
AUDIO_CHUNKS_FOLDER = r"\\jon-nas\jon-nas\videos\livestream_videos\2025-08-07-JSM-Livestream\chunks"


# audio_file_path, audio_chunk_paths = extract_audio_from_video(VIDEO_PATH)

asyncio.run(transcribe_audio_chunk_folder(AUDIO_CHUNKS_FOLDER))
