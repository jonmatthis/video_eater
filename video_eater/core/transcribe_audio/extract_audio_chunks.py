import math
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from tqdm import tqdm


def extract_single_chunk(args):
    """Extract a single audio chunk."""
    input_file, start_time, end_time, chunk_path = args

    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-ss", str(start_time),
        "-to", str(end_time),
        "-q:a", "0",
        "-y", chunk_path,
    ]

    subprocess.run(
        cmd, check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return chunk_path


def extract_audio_from_video(
        video_file: str,
        audio_output_file: str | None = None,
        chunk_audio: bool = True,
        audio_chunk_folder: str | None = None,
        chunk_length_seconds: float = 600,
        chunk_overlap_seconds: float = 15,
) -> tuple[str, List[str]]:
    """
    Extract audio from a video file and optionally chunk it into overlapping segments.

    Args:
        video_file: Path to the input video file
        audio_output_file: Path to save the extracted audio (defaults to video filename with .wav extension)
        chunk_audio: Whether to split the audio into chunks
        audio_chunk_folder: Folder to save audio chunks
        chunk_length_seconds: Length of each chunk in seconds
        chunk_overlap_seconds: Overlap between consecutive chunks in seconds

    Returns:
        Tuple of (path to extracted audio file, list of chunk file paths)
    """
    # If output file is not specified, create one based on input filename with .wav extension
    if audio_output_file is None:
        audio_output_file = video_file.replace(Path(video_file).suffix, ".wav")

    if not Path(audio_output_file).is_file():
        # Use ffmpeg directly for audio extraction - save as WAV format
        cmd = [
            "ffmpeg",
            "-i",
            video_file,
            "-vn",  # Disable video
            "-ac", "1",  # Set audio channels to 1 (mono)
            "-ar", "16000",  # Set audio sample rate to 16kHz
            "-y",  # Overwrite output file if it exists
            audio_output_file,
        ]

        print(f"Extracting audio from {Path(video_file).name} to {audio_output_file}")
        subprocess.run(cmd, check=True)

    # Chunk the audio if requested (chunks will still be MP3)
    chunk_files = []
    if chunk_audio:
        if audio_chunk_folder is not None:
            chunk_folder = Path(audio_chunk_folder)
            chunk_folder.mkdir(parents=True, exist_ok=True)
        else:
            chunk_folder = Path(audio_output_file).parent
        chunk_files = chunk_audio_file(
            input_file=audio_output_file,
            audio_chunk_folder=str(chunk_folder),
            chunk_length_seconds=chunk_length_seconds,
            chunk_overlap_seconds=chunk_overlap_seconds,
        )

    return audio_output_file, chunk_files


def chunk_audio_file(
        input_file: str,
        audio_chunk_folder: str,
        chunk_length_seconds: float = 600,
        chunk_overlap_seconds: float = 30,
        max_workers: int = 10,  # Limit parallel workers
        min_last_chunk_seconds: float = 60,  # Minimum duration for last chunk
) -> List[str]:
    """
    Split an audio file into overlapping chunks using ffmpeg.

    Handles short files by creating a single chunk if duration < chunk_length.
    Combines last two chunks if the final chunk would be too short.

    Args:
        input_file: Path to the input audio file
        audio_chunk_folder: Folder to save chunks
        chunk_length_seconds: Length of each chunk in seconds
        chunk_overlap_seconds: Overlap between consecutive chunks in seconds
        max_workers: Maximum number of parallel workers for extraction
        min_last_chunk_seconds: Minimum duration for the last chunk (will combine with previous if shorter)

    Returns:
        List of paths to the created chunk files
    """

    input_path = Path(input_file)
    base_name = input_path.stem

    # Get audio duration using ffprobe
    duration_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        input_file,
    ]
    duration = float(subprocess.check_output(duration_cmd).decode().strip())

    # Handle short videos: if duration is less than or equal to chunk_length, create single chunk
    if duration <= chunk_length_seconds:
        print(f"Audio duration ({duration:.1f}s) <= chunk length ({chunk_length_seconds}s), creating single chunk")
        chunk_filename = f"{base_name}_chunk_000__00.00sec.mp3"
        chunk_path = str(Path(audio_chunk_folder) / chunk_filename)

        # Extract the entire audio as a single chunk
        extract_single_chunk((input_file, 0, duration, chunk_path))
        return [chunk_path]

    # Calculate the number of chunks for longer audio
    step_size = chunk_length_seconds - chunk_overlap_seconds
    if step_size <= 0:
        raise ValueError("Overlap must be less than chunk length")

    # Calculate initial number of chunks
    num_chunks = math.ceil((duration - chunk_overlap_seconds) / step_size)

    # Check if the last chunk would be too short
    # Calculate where the last chunk would start
    last_chunk_start = (num_chunks - 1) * step_size
    last_chunk_duration = duration - last_chunk_start

    # If last chunk is too short, we'll combine it with the previous chunk
    combine_last_chunks = (num_chunks > 1 and
                           last_chunk_duration < min_last_chunk_seconds)

    if combine_last_chunks:
        print(f"Last chunk would be {last_chunk_duration:.1f}s, combining with previous chunk")
        effective_num_chunks = num_chunks - 1
    else:
        effective_num_chunks = num_chunks

    # Prepare all chunk tasks
    chunk_tasks = []

    for i in range(effective_num_chunks):
        # Calculate start time for this chunk
        start_time = i * step_size

        # Calculate end time
        if i == effective_num_chunks - 1:
            # This is the last chunk - extend to end of file
            end_time = duration
        else:
            # Regular chunk
            end_time = min(start_time + chunk_length_seconds, duration)

        # Create output filename
        chunk_filename = f"{base_name}_chunk_{i:03d}__{start_time:.1f}sec.mp3"
        chunk_path = str(Path(audio_chunk_folder) / chunk_filename)

        chunk_tasks.append((input_file, start_time, end_time, chunk_path))

    # Extract chunks in parallel
    chunk_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_single_chunk, task): task
                   for task in chunk_tasks}

        with tqdm(total=len(futures), desc="Chunking audio", unit="chunk") as pbar:
            for future in as_completed(futures):
                chunk_path = future.result()
                chunk_files.append(chunk_path)
                pbar.update(1)

    # Sort by filename to maintain order
    return sorted(chunk_files)


if __name__ == "__main__":
    # Example usage
    VIDEO_PATH = r"\\jon-nas\jon-nas\videos\livestream_videos\2025-08-07-JSM-Livestream\2025-08-07-JSM-Livestream-RAW.mp4"
    _audio_path, _chunk_paths = extract_audio_from_video(VIDEO_PATH)

    # Print chunk information
    print(f"\nCreated {len(_chunk_paths)} chunks:")
    for chunk in _chunk_paths:
        print(f"  - {Path(chunk).name}")
