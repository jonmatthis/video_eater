import math
import subprocess
from pathlib import Path
from typing import List

from tqdm import tqdm


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
        audio_output_file: Path to save the extracted audio (defaults to video filename with .mp3 extension)
        chunk_audio: Whether to split the audio into chunks
        chunk_length_seconds: Length of each chunk in seconds
        chunk_overlap_seconds: Overlap between consecutive chunks in seconds

    Returns:
        Path to the extracted audio file
    """
    # If output file is not specified, create one based on input filename
    if audio_output_file is None:
        audio_output_file = video_file.replace(Path(video_file).suffix, ".mp3")

    if not Path(audio_output_file).is_file():
        # Use ffmpeg directly for audio extraction - much simpler and more reliable
        cmd = [
            "ffmpeg",
            "-i",
            video_file,
            "-q:a",
            "0",  # Use highest quality
            "-map",
            "a",  # Extract only audio
            "-y",  # Overwrite output file if it exists
            audio_output_file,
        ]

        print(f"Extracting audio from {Path(video_file).name} to {audio_output_file}")
        subprocess.run(cmd, check=True)

    # Chunk the audio if requested
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
) -> List[str]:
    """
    Split an audio file into overlapping chunks using ffmpeg.

    Args:
        input_file: Path to the input audio file
        chunk_length_seconds: Length of each chunk in seconds
        chunk_overlap_seconds: Overlap between consecutive chunks in seconds

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

    # Calculate the number of chunks
    step_size = chunk_length_seconds - chunk_overlap_seconds
    num_chunks = math.ceil(duration / step_size) if step_size > 0 else 1

    chunk_files = []

    # Create chunks with progress bar
    with tqdm(total=num_chunks, desc=f"Chunking audio", unit="chunk") as pbar:
        for i in range(num_chunks):
            # Calculate start and end times for this chunk
            start_time = i * step_size
            end_time = min(start_time + chunk_length_seconds, duration)
            hours = int(start_time // 3600)
            minutes = int((start_time % 3600) // 60)
            seconds = int(start_time % 60)
            # Create output filename
            chunk_filename = f"{base_name}_chunk_{i:03d}_{hours:02d}h-{minutes:02d}m-{seconds:02d}sec.mp3"
            chunk_path = str(Path(audio_chunk_folder) / chunk_filename)
            chunk_files.append(chunk_path)

            # Use ffmpeg to extract the chunk
            cmd = [
                "ffmpeg",
                "-i",
                input_file,
                "-ss",
                str(start_time),
                "-to",
                str(end_time),
                "-q:a",
                "0",  # Use highest quality
                "-y",  # Overwrite output file if it exists
                chunk_path,
            ]

            # Run the command
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )

            # Update progress
            pbar.update(1)
            pbar.set_description(
                f"Chunk {i + 1}/{num_chunks}: {start_time:.1f}s-{end_time:.1f}s"
            )

    print(f"Created {len(chunk_files)} audio chunks in {audio_chunk_folder}")
    return chunk_files


if __name__ == "__main__":
    # Example usage
    VIDEO_PATH = r"\\jon-nas\jon-nas\videos\livestream_videos\2025-08-07-JSM-Livestream\2025-08-07-JSM-Livestream-RAW.mp4"
    _audio_path, _chunk_paths = extract_audio_from_video(VIDEO_PATH)
