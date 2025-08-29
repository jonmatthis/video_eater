from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import subprocess
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


def chunk_audio_file_parallel(
        input_file: str,
        audio_chunk_folder: str,
        chunk_length_seconds: float = 600,
        chunk_overlap_seconds: float = 30,
        max_workers: int = 4,  # Limit parallel workers
) -> List[str]:
    """Parallel version of chunk_audio_file."""

    # ... (duration calculation same as before) ...

    # Prepare all chunk tasks
    chunk_tasks = []
    for i in range(num_chunks - 1):
        start_time = i * step_size
        end_time = min(start_time + chunk_length_seconds, duration)
        # ... (filename generation same as before) ...
        chunk_tasks.append((input_file, start_time, end_time, chunk_path))

    chunk_files = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_single_chunk, task): task
                   for task in chunk_tasks}

        with tqdm(total=len(futures), desc="Chunking audio", unit="chunk") as pbar:
            for future in as_completed(futures):
                chunk_path = future.result()
                chunk_files.append(chunk_path)
                pbar.update(1)

    return sorted(chunk_files)  # Sort to maintain order