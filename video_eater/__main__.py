import sys
import asyncio
from pathlib import Path
import time
from datetime import datetime

from video_eater.core.ai_processors.transcript_processor import TranscriptProcessor

# Add the parent directory to sys.path to make the package importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from video_eater.core.transcribe_audio.transcribe_audio_chunks import transcribe_audio_chunk_folder
from video_eater.core.transcribe_audio.extract_audio_chunks import extract_audio_from_video

# Configuration flags
RE_CHUNK_AUDIO = False
RE_TRANSCRIBE_AUDIO = True or RE_CHUNK_AUDIO
RE_ANALYZE_TRANSCRIPTS = False or RE_TRANSCRIBE_AUDIO

# Processing configuration
PROCESSING_CONFIG = {
    'max_concurrent_chunks': 50,  # Max simultaneous chunk analyses
    'batch_size': 10,  # Process chunks in batches
    'chunk_length_seconds': 600,  # Audio chunk length (10 minutes)
    'chunk_overlap_seconds': 15,  # Audio chunk overlap
    'model': 'deepseek-chat',  # AI model to use
}

# Video path
VIDEO_PATH = r"\\jon-nas\jon-nas\videos\livestream_videos\2025-08-14-JSM-Livestream-Skellycam\2025-08-14-JSM-Livestream-Skellycam.mp4"

# Folder setup
recording_folder = Path(VIDEO_PATH).parent
AUDIO_CHUNKS_FOLDER = recording_folder / "audio_chunks"
TRANSCRIPT_CHUNKS_FOLDER = recording_folder / "chunk_transcripts"
ANALYSIS_FOLDER = recording_folder / "analysis"


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    line = char * 60
    print(f"\n{line}")
    print(f"{text}")
    print(f"{line}")


def print_step(step_num: int, total: int, description: str):
    """Print a formatted step indicator."""
    print(f"\n[Step {step_num}/{total}] {description}")
    print("-" * 40)


async def process_audio_extraction(video_path: Path, audio_chunks_folder: Path) -> tuple:
    """Extract and chunk audio from video."""
    print_step(1, 3, "Audio Extraction & Chunking")

    if not audio_chunks_folder.exists() or RE_CHUNK_AUDIO:
        print(f"📹 Extracting audio from: {video_path.name}")
        audio_chunks_folder.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        audio_file_path, audio_chunk_paths = extract_audio_from_video(
            video_file=str(video_path),
            audio_chunk_folder=str(audio_chunks_folder),
            chunk_length_seconds=PROCESSING_CONFIG['chunk_length_seconds'],
            chunk_overlap_seconds=PROCESSING_CONFIG['chunk_overlap_seconds']
        )
        elapsed = time.time() - start_time

        print(f"✅ Created {len(audio_chunk_paths)} audio chunks in {elapsed:.1f}s")
        print(f"   • Chunk length: {PROCESSING_CONFIG['chunk_length_seconds']}s")
        print(f"   • Overlap: {PROCESSING_CONFIG['chunk_overlap_seconds']}s")
        return audio_file_path, audio_chunk_paths
    else:
        existing_chunks = list(audio_chunks_folder.glob("*.mp3"))
        print(f"📂 Using existing {len(existing_chunks)} audio chunks")
        return None, existing_chunks


async def process_transcription(audio_chunks_folder: Path, transcript_chunks_folder: Path) -> list:
    """Transcribe audio chunks."""
    print_step(2, 3, "Audio Transcription")

    # Count existing files
    audio_files = list(audio_chunks_folder.glob("*.mp3"))
    transcript_files = []
    if transcript_chunks_folder.exists():
        transcript_files = list(transcript_chunks_folder.glob("*.json"))

    print(f"📊 Audio chunks: {len(audio_files)}")
    print(f"📊 Existing transcripts: {len(transcript_files)}")

    needs_transcription = (
            not transcript_chunks_folder.exists() or
            len(audio_files) > len(transcript_files) or
            RE_TRANSCRIBE_AUDIO
    )

    if needs_transcription:
        print(f"🎙️ Starting transcription process...")
        start_time = time.time()

        transcripts = await transcribe_audio_chunk_folder(
            chunk_folder=str(audio_chunks_folder),
            re_transcribe=RE_TRANSCRIBE_AUDIO
        )

        elapsed = time.time() - start_time
        print(f"✅ Transcribed {len(transcripts)} chunks in {elapsed:.1f}s")
        print(f"   • Average time per chunk: {elapsed / len(transcripts):.1f}s")
        return transcripts
    else:
        print(f"📂 Using existing transcripts")
        return transcript_files


async def process_analysis(transcript_folder: Path, analysis_folder: Path):
    """Analyze transcripts and generate summaries."""
    print_step(3, 3, "Transcript Analysis & Summary Generation")

    full_analysis_file = analysis_folder / "full_video_analysis.yaml"

    if not full_analysis_file.exists() or RE_ANALYZE_TRANSCRIPTS:
        print(f"🧠 Starting AI-powered analysis...")
        print(f"   • Model: {PROCESSING_CONFIG['model']}")
        print(f"   • Max concurrent: {PROCESSING_CONFIG['max_concurrent_chunks']}")
        print(f"   • Batch size: {PROCESSING_CONFIG['batch_size']}")

        # Initialize processor with async configuration
        processor = TranscriptProcessor(
            model=PROCESSING_CONFIG['model'],
            use_async=True,
            max_concurrent_chunks=PROCESSING_CONFIG['max_concurrent_chunks'],
            batch_size=PROCESSING_CONFIG['batch_size']
        )

        start_time = time.time()

        # Process all transcripts with async parallelization
        full_analysis = await processor.process_transcript_folder(
            transcript_folder=transcript_folder,
            output_folder=analysis_folder
        )

        elapsed = time.time() - start_time

        # Display results
        print_header("VIDEO ANALYSIS COMPLETE! 🎉")

        print(f"\n⏱️ Total processing time: {elapsed:.1f}s")
        print(f"\n📝 Executive Summary:")
        print(f"   {full_analysis.executive_summary}\n")

        print(f"📊 Analysis Statistics:")
        print(f"   • {len(full_analysis.chapters)} chapters generated")
        print(f"   • {len(full_analysis.main_topics)} main topics identified")
        print(f"   • {len(full_analysis.key_takeaways)} key takeaways extracted")

        if full_analysis.notable_quotes:
            print(f"   • {len(full_analysis.notable_quotes)} notable quotes collected")

        print(f"\n📁 Output Files:")
        print(f"   Location: {analysis_folder}")
        print(f"   • full_video_analysis.yaml - Complete analysis")
        print(f"   • youtube_description.txt - Ready for YouTube")
        print(f"   • video_analysis_report.md - Detailed markdown report")
        print(f"   • chunk_*.analysis.yaml - Individual chunk analyses")

        return full_analysis
    else:
        print(f"📂 Analysis already exists: {full_analysis_file}")
        print(f"   Set RE_ANALYZE_TRANSCRIPTS=True to regenerate")
        return None


async def main():
    """Run the complete video processing pipeline."""

    total_start = time.time()

    print_header(f"VIDEO PROCESSING PIPELINE")
    print(f"📹 Video: {Path(VIDEO_PATH).name}")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Step 1: Extract and chunk audio
        await process_audio_extraction(Path(VIDEO_PATH), AUDIO_CHUNKS_FOLDER)

        # Step 2: Transcribe audio chunks
        await process_transcription(AUDIO_CHUNKS_FOLDER, TRANSCRIPT_CHUNKS_FOLDER)

        # Step 3: Analyze transcripts
        await process_analysis(TRANSCRIPT_CHUNKS_FOLDER, ANALYSIS_FOLDER)

        # Final summary
        total_elapsed = time.time() - total_start
        print_header("✅ PIPELINE COMPLETE!")
        print(f"⏱️ Total pipeline time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} minutes)")
        print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user")
        raise
    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    # Configure async event loop for better performance
    if sys.platform == 'win32':
        # Windows-specific event loop policy for better performance
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)