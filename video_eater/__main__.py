import sys
import asyncio
from pathlib import Path

from video_eater.core.ai_processors.transcript_processor import TranscriptProcessor

# Add the parent directory to sys.path to make the package importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from video_eater.core.transcribe_audio.transcribe_audio_chunks import transcribe_audio_chunk_folder
from video_eater.core.transcribe_audio.extract_audio_chunks import extract_audio_from_video

# Configuration flags
RE_CHUNK_AUDIO = False
RE_TRANSCRIBE_AUDIO = True or RE_CHUNK_AUDIO
RE_ANALYZE_TRANSCRIPTS = False  or RE_TRANSCRIBE_AUDIO# New flag for analysis

# Video path
VIDEO_PATH = r"\\jon-nas\jon-nas\videos\livestream_videos\2025-08-14-JSM-Livestream-Skellycam\2025-08-14-JSM-Livestream-Skellycam.mp4"

# Folder setup
recording_folder = Path(VIDEO_PATH).parent
AUDIO_CHUNKS_FOLDER = recording_folder / "audio_chunks"
TRANSCRIPT_CHUNKS_FOLDER = recording_folder / "chunk_transcripts"
ANALYSIS_FOLDER = recording_folder / "analysis"


async def main():
    """Run the complete video processing pipeline."""

    # Step 1: Extract and chunk audio
    if not AUDIO_CHUNKS_FOLDER.exists() or RE_CHUNK_AUDIO:
        print(f"Extracting audio from video: {VIDEO_PATH}")
        AUDIO_CHUNKS_FOLDER.mkdir(parents=True, exist_ok=True)
        audio_file_path, audio_chunk_paths = extract_audio_from_video(
            video_file=VIDEO_PATH,
            audio_chunk_folder=str(AUDIO_CHUNKS_FOLDER),
            chunk_length_seconds=600,  # 10-minute chunks
            chunk_overlap_seconds=15  # 15-second overlap
        )
        print(f"Created {len(audio_chunk_paths)} audio chunks")
    else:
        print(f"Using existing audio chunks from: {AUDIO_CHUNKS_FOLDER}")

    # Step 2: Transcribe audio chunks
    number_of_audio_chunks = len([*AUDIO_CHUNKS_FOLDER.glob(".mp3")])
    number_of_transcript_chunks = 0
    if TRANSCRIPT_CHUNKS_FOLDER.exists():
        number_of_transcript_chunks = len([*AUDIO_CHUNKS_FOLDER.glob(".json")])

    if not TRANSCRIPT_CHUNKS_FOLDER.exists() or (number_of_audio_chunks>number_of_transcript_chunks) or RE_TRANSCRIBE_AUDIO:
        print(f"Transcribing audio chunks in folder: {AUDIO_CHUNKS_FOLDER}")
        transcripts = await transcribe_audio_chunk_folder(
            chunk_folder=str(AUDIO_CHUNKS_FOLDER),
            re_transcribe=RE_TRANSCRIBE_AUDIO
        )
        print(f"Completed transcription of {len(transcripts)} chunks")
    else:
        print(f"Using existing transcripts from: {TRANSCRIPT_CHUNKS_FOLDER}")

    # Step 3: Analyze transcripts and generate summaries
    full_analysis_file = ANALYSIS_FOLDER / "full_video_analysis.yaml"
    if not full_analysis_file.exists() or RE_ANALYZE_TRANSCRIPTS:
        print(f"Analyzing transcripts in folder: {TRANSCRIPT_CHUNKS_FOLDER}")

        # Initialize the processor
        processor = TranscriptProcessor(use_async=True)

        # Process all transcripts
        video_title = Path(VIDEO_PATH).stem  # Extract video title from filename
        full_analysis = await processor.process_transcript_folder(
            transcript_folder=TRANSCRIPT_CHUNKS_FOLDER,
            output_folder=ANALYSIS_FOLDER
        )

        print("\n" + "=" * 60)
        print("VIDEO ANALYSIS COMPLETE!")
        print("=" * 60)
        print(f"\n📝 Executive Summary:")
        print(f"{full_analysis.executive_summary}\n")

        print(f"📚 Generated {len(full_analysis.chapters)} chapters")
        print(f"🎯 Found {len(full_analysis.main_topics)} main topics")
        print(f"💡 Extracted {len(full_analysis.key_takeaways)} key takeaways")

        if full_analysis.notable_quotes:
            print(f"💬 Collected {len(full_analysis.notable_quotes)} notable quotes")

        print(f"\n📁 Output files saved to: {ANALYSIS_FOLDER}")
        print("  - full_video_analysis.yaml (complete analysis)")
        print("  - youtube_description.txt (ready for YouTube)")
        print("  - video_analysis_report.md (detailed markdown report)")
        print("  - Individual chunk analyses (*.analysis.yaml)")
    else:
        print(f"Analysis already exists at: {full_analysis_file}")
        print("Set RE_ANALYZE_TRANSCRIPTS=True to regenerate")

    print("\n✅ Pipeline complete!")


if __name__ == "__main__":
    asyncio.run(main())