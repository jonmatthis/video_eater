import sys
import asyncio
from pathlib import Path
import time
from datetime import datetime
from typing import Optional, List, Tuple

# Add the parent directory to sys.path to make the package importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from video_eater.core.ai_processors.transcript_processor import TranscriptProcessor
from video_eater.core.transcribe_audio.transcribe_audio_chunks import transcribe_audio_chunk_folder
from video_eater.core.transcribe_audio.extract_audio_chunks import extract_audio_from_video


class VideoProcessingPipeline:
    """Manages the complete video processing pipeline with proper caching."""

    def __init__(self,
                 video_path: str,
                 force_chunk_audio: bool = False,
                 force_transcribe: bool = False,
                 force_analyze: bool = False,
                 processing_config: dict = None):
        """
        Initialize the video processing pipeline.

        Args:
            video_path: Path to the video file
            force_chunk_audio: Force re-chunking of audio even if chunks exist
            force_transcribe: Force re-transcription even if transcripts exist
            force_analyze: Force re-analysis even if analysis exists
            processing_config: Configuration dictionary for processing parameters
        """
        self.video_path = Path(video_path)
        self.force_chunk_audio = force_chunk_audio
        self.force_transcribe = force_transcribe
        self.force_analyze = force_analyze

        # Default configuration
        self.config = {
            'max_concurrent_chunks': 50,
            'batch_size': 50,
            'chunk_length_seconds': 600,
            'chunk_overlap_seconds': 30,
            # 'model': 'deepseek-chat',
            'model': 'gpt-4.1',
        }

        if processing_config:
            self.config.update(processing_config)

        # Set up folder structure
        self.recording_folder = self.video_path.parent
        self.audio_chunks_folder = self.recording_folder / "audio_chunks"
        self.transcript_chunks_folder = self.recording_folder / "transcript_chunks"
        self.analysis_folder = self.recording_folder / "analysis_chunks"

        # Track what was processed
        self.processing_stats = {
            'audio_chunked': False,
            'audio_transcribed': False,
            'analysis_completed': False,
            'used_cache': {'audio': False, 'transcripts': False, 'analysis': False}
        }

    def print_header(self, text: str, char: str = "="):
        """Print a formatted header."""
        line = char * 60
        print(f"\n{line}")
        print(f"{text}")
        print(f"{line}")

    def print_step(self, step_num: int, total: int, description: str):
        """Print a formatted step indicator."""
        print(f"\n[Step {step_num}/{total}] {description}")
        print("-" * 40)

    def check_audio_chunks_exist(self) -> bool:
        """Check if audio chunks already exist."""
        if not self.audio_chunks_folder.exists():
            return False
        chunks = list(self.audio_chunks_folder.glob("*.mp3"))
        return len(chunks) > 0

    def check_transcripts_exist(self) -> Tuple[bool, int, int]:
        """
        Check if transcripts exist and return status.

        Returns:
            Tuple of (all_exist, num_audio_chunks, num_transcripts)
        """
        if not self.transcript_chunks_folder.exists():
            return False, 0, 0

        audio_chunks = list(self.audio_chunks_folder.glob("*.mp3")) if self.audio_chunks_folder.exists() else []
        transcripts = list(self.transcript_chunks_folder.glob("*.transcript.json"))

        # Check if we have a transcript for each audio chunk
        all_exist = len(audio_chunks) > 0 and len(transcripts) >= len(audio_chunks)
        return all_exist, len(audio_chunks), len(transcripts)

    def check_analysis_exists(self) -> bool:
        """Check if full analysis already exists."""
        if not self.analysis_folder.exists():
            return False
        full_analysis_file = self.analysis_folder / "full_video_analysis.yaml"
        return full_analysis_file.exists()

    async def process_audio_extraction(self) -> Tuple[Optional[str], List[str]]:
        """Extract and chunk audio from video."""
        self.print_step(1, 3, "Audio Extraction & Chunking")

        # Check if we should use existing chunks
        if self.check_audio_chunks_exist() and not self.force_chunk_audio:
            existing_chunks = sorted(list(self.audio_chunks_folder.glob("*.mp3")))
            print(f"📂 Using existing {len(existing_chunks)} audio chunks")
            print(f"   ℹ️ Use --force-chunk-audio to re-extract")
            self.processing_stats['used_cache']['audio'] = True
            return None, [str(p) for p in existing_chunks]

        # Extract and chunk audio
        print(f"📹 Extracting audio from: {self.video_path.name}")
        if self.force_chunk_audio and self.audio_chunks_folder.exists():
            print(f"   ⚠️ Force re-chunking enabled, removing existing chunks...")
            for chunk in self.audio_chunks_folder.glob("*.mp3"):
                chunk.unlink()

        self.audio_chunks_folder.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        audio_file_path, audio_chunk_paths = extract_audio_from_video(
            video_file=str(self.video_path),
            audio_chunk_folder=str(self.audio_chunks_folder),
            chunk_length_seconds=self.config['chunk_length_seconds'],
            chunk_overlap_seconds=self.config['chunk_overlap_seconds']
        )
        elapsed = time.time() - start_time

        print(f"✅ Created {len(audio_chunk_paths)} audio chunks in {elapsed:.1f}s")
        print(f"   • Chunk length: {self.config['chunk_length_seconds']}s")
        print(f"   • Overlap: {self.config['chunk_overlap_seconds']}s")
        self.processing_stats['audio_chunked'] = True
        return audio_file_path, audio_chunk_paths

    async def process_transcription(self, local_whisper:bool=False, use_assembly_ai:bool=True, ) -> List[Path]:
        """Transcribe audio chunks."""
        self.print_step(2, 3, "Audio Transcription")

        # Check existing transcripts
        all_exist, num_audio, num_transcripts = self.check_transcripts_exist()

        print(f"📊 Audio chunks: {num_audio}")
        print(f"📊 Existing transcripts: {num_transcripts}")

        if all_exist and not self.force_transcribe:
            print(f"📂 Using existing transcripts (all {num_audio} chunks have transcripts)")
            print(f"   ℹ️ Use --force-transcribe to re-transcribe")
            self.processing_stats['used_cache']['transcripts'] = True
            return sorted(list(self.transcript_chunks_folder.glob("*.transcript.json")))

        # Transcribe missing or all chunks
        if self.force_transcribe:
            print(f"   ⚠️ Force transcribe enabled, re-transcribing all chunks...")
        elif num_transcripts < num_audio:
            print(f"   ℹ️ Missing {num_audio - num_transcripts} transcripts, transcribing...")

        print(f"🎙️ Starting transcription process...")
        start_time = time.time()

        await transcribe_audio_chunk_folder(
            audio_chunk_folder=str(self.audio_chunks_folder),
            transcript_chunk_folder=str(self.transcript_chunks_folder),
            local_whisper=local_whisper,
            use_assembly_ai=use_assembly_ai,
            re_transcribe=self.force_transcribe
        )

        elapsed = time.time() - start_time

        # Get final transcript count
        transcripts = sorted(list(self.transcript_chunks_folder.glob("*.transcript.json")))
        new_transcripts = transcripts if self.force_transcribe else num_audio - num_transcripts

        print(f"✅ Transcription complete in {elapsed:.1f}s")
        if new_transcripts:
            print(f"   • Transcribed {new_transcripts} chunks")
            print(f"   • Average time per chunk: {elapsed / new_transcripts:.1f}s")

        self.processing_stats['audio_transcribed'] = True
        return transcripts

    async def process_analysis(self):
        """Analyze transcripts and generate summaries."""
        self.print_step(3, 3, "Transcript Analysis & Summary Generation")

        if self.check_analysis_exists() and not self.force_analyze:
            print(f"📂 Analysis already exists")
            print(f"   ℹ️ Use --force-analyze to regenerate")
            self.processing_stats['used_cache']['analysis'] = True

            # Load and display existing analysis summary
            import yaml
            analysis_file = self.recording_folder / "full_video_analysis.yaml"
            with open(analysis_file, 'r', encoding='utf-8') as f:
                analysis = yaml.safe_load(f)

            print(f"\n📝 Executive Summary:")
            print(f"   {analysis['executive_summary']}")
            print(f"\n📊 Analysis contains:")
            print(f"   • {len(analysis.get('chapters', []))} chapters")
            print(f"   • {len(analysis.get('main_themes', analysis.get('main_topics', [])))} main topics")
            print(f"   • {len(analysis.get('key_takeaways', []))} key takeaways")
            return analysis

        if self.force_analyze:
            print(f"   ⚠️ Force analyze enabled, regenerating analysis...")

        print(f"🧠 Starting AI auto analysis...")
        print(f"   • Model: {self.config['model']}")
        print(f"   • Max concurrent: {self.config['max_concurrent_chunks']}")
        print(f"   • Batch size: {self.config['batch_size']}")

        # Initialize processor
        processor = TranscriptProcessor(
            model=self.config['model'],
            use_async=True,
            max_concurrent_chunks=self.config['max_concurrent_chunks'],
            batch_size=self.config['batch_size'],
            chunk_length_seconds=self.config['chunk_length_seconds'],
            chunk_overlap_seconds=self.config['chunk_overlap_seconds']
        )

        start_time = time.time()

        # Process all transcripts
        full_analysis = await processor.process_transcript_folder(
            transcript_folder=self.transcript_chunks_folder,
            chunk_analysis_output_folder=self.recording_folder / "analysis_chunks"
        )

        elapsed = time.time() - start_time

        # Display results
        self.print_header("VIDEO ANALYSIS COMPLETE! 🎉")

        print(f"\n⏱️ Total processing time: {elapsed:.1f}s")
        print(f"\n📝 Executive Summary:")
        print(f"   {full_analysis.executive_summary}\n")
        print(f"📊 Analysis Statistics:")
        print(f"   • {len(full_analysis.chapters)} chapters generated")
        print(f"   • {len(full_analysis.main_themes)} main topics identified")
        print(f"   • {len(full_analysis.key_takeaways)} key takeaways extracted")

        if full_analysis.pull_quotes:
            print(f"   • {len(full_analysis.pull_quotes)} notable quotes collected")

        print(f"\n📁 Output Files:")
        print(f"   Location: {self.analysis_folder}")
        print(f"   • full_video_analysis.yaml - Complete analysis")
        print(f"   • youtube_description.txt - Ready for YouTube")
        print(f"   • video_analysis_report.md - Detailed markdown report")
        print(f"   • chunk_*.analysis.yaml - Individual chunk analyses")

        self.processing_stats['analysis_completed'] = True
        return full_analysis

    async def run(self):
        """Run the complete video processing pipeline."""
        total_start = time.time()

        self.print_header("VIDEO PROCESSING PIPELINE")
        print(f"📹 Video: {self.video_path.name}")
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Print configuration
        print(f"\n⚙️ Configuration:")
        print(f"   • Force chunk audio: {self.force_chunk_audio}")
        print(f"   • Force transcribe: {self.force_transcribe}")
        print(f"   • Force analyze: {self.force_analyze}")

        try:
            # Step 1: Extract and chunk audio
            await self.process_audio_extraction()

            # Step 2: Transcribe audio chunks
            await self.process_transcription(local_whisper=False, use_assembly_ai=True)

            # Step 3: Analyze transcripts
            await self.process_analysis()

            # Final summary
            total_elapsed = time.time() - total_start
            self.print_header("✅ PIPELINE COMPLETE!")
            print(f"⏱️ Total pipeline time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} minutes)")
            print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Print cache usage summary
            print(f"\n💾 Cache Usage:")
            for step, used in self.processing_stats['used_cache'].items():
                status = "✓ Used cache" if used else "⟲ Processed"
                print(f"   • {step.capitalize()}: {status}")

        except KeyboardInterrupt:
            print("\n\n⚠️ Pipeline interrupted by user")
            raise
        except Exception as e:
            print(f"\n\n❌ Pipeline failed with error: {e}")
            raise


async def main():
    """Main entry point with command line argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description='Process video for transcription and analysis')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--force-chunk-audio', action='store_true',
                        help='Force re-chunking of audio even if chunks exist')
    parser.add_argument('--force-transcribe', action='store_true',
                        help='Force re-transcription even if transcripts exist')
    parser.add_argument('--force-analyze', action='store_true',
                        help='Force re-analysis even if analysis exists')
    parser.add_argument('--force-all', action='store_true',
                        help='Force all processing steps')
    parser.add_argument('--model', default='deepseek-chat',
                        help='AI model to use for analysis')
    parser.add_argument('--chunk-length', type=int, default=600,
                        help='Audio chunk length in seconds (default: 600)')
    parser.add_argument('--chunk-overlap', type=int, default=15,
                        help='Audio chunk overlap in seconds (default: 15)')

    args = parser.parse_args()

    # Handle force-all flag
    force_chunk = args.force_chunk_audio or args.force_all
    force_transcribe = args.force_transcribe or args.force_all
    force_analyze = args.force_analyze or args.force_all

    # Create configuration
    config = {
        'model': args.model,
        'chunk_length_seconds': args.chunk_length,
        'chunk_overlap_seconds': args.chunk_overlap,
    }

    # Create and run pipeline
    pipeline = VideoProcessingPipeline(
        video_path=args.video_path,
        force_chunk_audio=force_chunk,
        force_transcribe=force_transcribe,
        force_analyze=force_analyze,
        processing_config=config
    )

    await pipeline.run()


if __name__ == "__main__":
    # Configure async event loop for better performance
    if sys.platform == 'win32':
        # Windows-specific event loop policy for better performance
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    try:
        # If no command line args provided, use default video path
        if len(sys.argv) == 1:
            # Default video for testing
            VIDEO_PATH = r"\\jon-nas\jon-nas\videos\livestream_videos\2025-08-14-JSM-Livestream-Skellycam\2025-08-14-JSM-Livestream-Skellycam.mp4"
            # VIDEO_PATH = r"\\jon-nas\jon-nas\videos\livestream_videos\2025-08-07-JSM-Livestream\2025-08-07-JSM-Livestream-RAW.mp4"

            _pipeline = VideoProcessingPipeline(
                video_path=VIDEO_PATH,
                force_chunk_audio=False,
                force_transcribe=False,
                force_analyze=False
            )
            asyncio.run(_pipeline.run())
        else:
            asyncio.run(main())

    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)