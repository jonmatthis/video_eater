import asyncio
import json
import yaml
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta
import re
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import time

from video_eater.core.ai_processors.base_processor import BaseAIProcessor
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChapterHeading(BaseModel):
    timestamp_seconds: float = Field(description="Start time in seconds")
    title: str = Field(description="Chapter title")
    description: str | None = Field(description="Brief description of what happens in this chapter")


class ChunkAnalysis(BaseModel):
    summary: str = Field(description="Comprehensive summary of the chunk content")
    key_topics: list[str] = Field(description="list of main topics discussed")
    topic_outline: dict[str, list[str] | dict[str, object]] = Field(
        description="Hierarchical outline of topics and subtopics")
    chapters: list[ChapterHeading] = Field(description="Timestamped chapter headings")
    notable_quotes: list[str] | None = Field(default=None, description="Important or interesting quotes")


class FullVideoAnalysis(BaseModel):
    executive_summary: str = Field(description="High-level summary of the entire video")
    detailed_summary: str = Field(description="Comprehensive summary with key points")
    main_topics: list[str] = Field(description="Primary topics covered across the video")
    complete_outline: dict[str, list[str]] = Field(description="Complete hierarchical topic outline")
    chapters: list[ChapterHeading] = Field(description="Full video chapter list with adjusted timestamps")
    key_takeaways: list[str] = Field(description="Main insights and conclusions")
    notable_quotes: list[str] | None = Field(default=None, description="Most impactful quotes from the video")


class TranscriptProcessor:
    """Process transcribed chunks to generate summaries, outlines, and chapters."""

    def __init__(self,
                 model: str = "deepseek-chat",
                 use_async: bool = True,
                 max_concurrent_chunks: int = 50,
                 batch_size: int = 10,
                 chunk_length_seconds: int = 600,  # Store chunk length
                 chunk_overlap_seconds: int = 15):  # Store overlap
        self.processor = BaseAIProcessor(model=model, use_async=use_async)
        self.model = model
        self.max_concurrent_chunks = max_concurrent_chunks
        self.batch_size = batch_size
        self.chunk_length_seconds = chunk_length_seconds
        self.chunk_overlap_seconds = chunk_overlap_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent_chunks)
        self.processing_stats = {
            'chunks_processed': 0,
            'chunks_cached': 0,
            'processing_time': 0,
            'errors': []
        }

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to YouTube-style timestamp (HH:MM:SS or MM:SS)."""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    def calculate_chunk_start_time(self, chunk_index: int) -> float:
        """Calculate the actual start time of a chunk accounting for overlap."""
        # First chunk starts at 0
        if chunk_index == 0:
            return 0.0

        # Subsequent chunks start at (chunk_length - overlap) * index
        effective_chunk_duration = self.chunk_length_seconds - self.chunk_overlap_seconds
        return chunk_index * effective_chunk_duration

    @staticmethod
    def parse_chunk_filename(filename: str) -> Tuple[int, float]:
        """Extract chunk index and start time from chunk filename.
        Returns: (chunk_index, start_time_seconds)
        """
        # Try to match chunk_INDEX_Xh-Ym-Zsec pattern
        pattern = r"chunk_(\d+)_(\d+)h-(\d+)m-(\d+)sec"
        match = re.search(pattern, filename)
        if match:
            chunk_index = int(match.group(1))
            hours = int(match.group(2))
            minutes = int(match.group(3))
            seconds = int(match.group(4))
            start_time = hours * 3600 + minutes * 60 + seconds
            return chunk_index, start_time

        # Fallback: try to get just the chunk index
        index_pattern = r"chunk_(\d+)"
        index_match = re.search(index_pattern, filename)
        if index_match:
            return int(index_match.group(1)), 0.0

        return 0, 0.0

    async def analyze_chunk_with_semaphore(self,
                                           transcript_text: str,
                                           chunk_start_seconds: float,
                                           chunk_index: int,
                                           chunk_name: str) -> Tuple[int, ChunkAnalysis, str]:
        """Analyze a single chunk with rate limiting via semaphore."""
        async with self._semaphore:
            try:
                start_time = time.time()
                print(
                    f"  📝 Processing chunk {chunk_index + 1}: {chunk_name} (starts at {self.format_timestamp(chunk_start_seconds)})")

                analysis = await self.analyze_chunk(
                    transcript_text=transcript_text,
                    chunk_start_seconds=chunk_start_seconds,
                    chunk_index=chunk_index
                )

                elapsed = time.time() - start_time
                print(f"  ✅ Completed chunk {chunk_index + 1} in {elapsed:.1f}s")

                return chunk_index, analysis, None

            except Exception as e:
                error_msg = f"Error analyzing chunk {chunk_index}: {str(e)}"
                print(f"  ❌ Failed chunk {chunk_index + 1}: {e}")
                self.processing_stats['errors'].append(error_msg)
                return chunk_index, None, error_msg

    async def analyze_chunk(self,
                            transcript_text: str,
                            chunk_start_seconds: float = 0,
                            chunk_index: int = 0) -> ChunkAnalysis:
        """Analyze a single transcript chunk."""

        system_prompt = f"""You are analyzing a transcript chunk from a longer video. 
        This is chunk #{chunk_index} starting at {self.format_timestamp(chunk_start_seconds)} in the video.

        Please provide:
        1. A comprehensive summary of this chunk
        2. Key topics discussed (as a list)
        3. A hierarchical topic outline (topics as keys, subtopics as lists)
        4. Timestamped chapter headings (3-7 chapters, depending on content)
        5. Notable pull-quotes if any

        For chapter timestamps, use relative times from the START of this chunk (0 seconds).
        Make chapters meaningful and descriptive, not just "Introduction" or "Conclusion".

        Transcript:
        {transcript_text}
        """

        try:
            response = await self.processor.async_make_openai_json_mode_ai_request(
                system_prompt=system_prompt,
                input_data={},
                output_model=ChunkAnalysis
            )

            # Adjust timestamps to account for chunk start time
            for chapter in response.chapters:
                chapter.timestamp_seconds += chunk_start_seconds

            return response

        except Exception as e:
            print(f"Error analyzing chunk {chunk_index}: {e}")
            raise

    async def _return_cached(self, idx: int, analysis: ChunkAnalysis):
        """Helper to return cached analysis in async context."""
        return idx, analysis, None

    async def process_transcript_batch(self,
                                       batch: List[Tuple[int, Path, float]],
                                       output_folder: Path) -> List[ChunkAnalysis]:
        """Process a batch of transcript files in parallel."""
        tasks = []

        for batch_idx, (chunk_idx, transcript_file, chunk_start) in enumerate(batch):
            # Check if analysis already exists
            analysis_file = output_folder / f"{transcript_file.stem}.analysis.yaml"

            if analysis_file.exists():
                print(f"  📂 Using cached analysis for chunk {chunk_idx + 1}")
                self.processing_stats['chunks_cached'] += 1
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    chunk_analysis = ChunkAnalysis(**yaml.safe_load(f))
                    tasks.append(asyncio.create_task(
                        self._return_cached(chunk_idx, chunk_analysis)
                    ))
            else:
                # Load transcript
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)

                # Extract text
                if isinstance(transcript_data, dict):
                    transcript_text = transcript_data.get('text', '')
                else:
                    transcript_text = str(transcript_data)

                # Create analysis task
                task = asyncio.create_task(
                    self.analyze_chunk_with_semaphore(
                        transcript_text=transcript_text,
                        chunk_start_seconds=chunk_start,
                        chunk_index=chunk_idx,
                        chunk_name=transcript_file.name
                    )
                )
                tasks.append(task)

        # Wait for all tasks in this batch to complete
        results = await asyncio.gather(*tasks)

        # Process results and save analyses
        chunk_analyses = []
        for idx, analysis, error in results:
            if error:
                continue  # Skip failed chunks

            chunk_analyses.append((idx, analysis))

            # Save successful analysis if it's new
            batch_item = next((item for item in batch if item[0] == idx), None)

            if batch_item:
                transcript_file = batch_item[1]
                analysis_file = output_folder / f"{transcript_file.stem}.analysis.yaml"

                if not analysis_file.exists():
                    with open(analysis_file, 'w', encoding='utf-8') as f:
                        yaml.dump(analysis.model_dump(), f,
                                  default_flow_style=False,
                                  sort_keys=False,
                                  allow_unicode=True)
                    self.processing_stats['chunks_processed'] += 1

        # Sort by index to maintain order
        chunk_analyses.sort(key=lambda x: x[0])
        return [analysis for _, analysis in chunk_analyses]

    async def process_transcript_folder(self,
                                        transcript_folder: Path,
                                        output_folder: Path | None = None) -> FullVideoAnalysis:
        """Process all transcript chunks in a folder with parallel processing."""

        start_time = time.time()

        # Set up output folder
        if output_folder is None:
            output_folder = transcript_folder.parent / "analysis_chunks"
        output_folder.mkdir(parents=True, exist_ok=True)

        # Find all transcript JSON files
        transcript_files = sorted(transcript_folder.glob("*.transcript.json"))
        if not transcript_files:
            raise ValueError(f"No transcript files found in {transcript_folder}")

        print(f"\n🎬 Processing {len(transcript_files)} transcript files")
        print(f"⚡ Max concurrent processing: {self.max_concurrent_chunks}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"⏱️ Chunk length: {self.chunk_length_seconds}s, Overlap: {self.chunk_overlap_seconds}s\n")

        # Prepare file data with corrected timestamps
        file_data = []
        for transcript_file in transcript_files:
            chunk_index, _ = self.parse_chunk_filename(transcript_file.name)
            # Calculate actual start time based on chunk index and overlap
            chunk_start = self.calculate_chunk_start_time(chunk_index)
            file_data.append((chunk_index, transcript_file, chunk_start))

        # Sort by chunk index to ensure proper ordering
        file_data.sort(key=lambda x: x[0])

        # Log calculated timestamps for verification
        print("📍 Calculated chunk start times:")
        for chunk_idx, file_path, start_time in file_data[:5]:  # Show first 5
            print(f"   Chunk {chunk_idx}: {self.format_timestamp(start_time)}")
        if len(file_data) > 5:
            last_chunk = file_data[-1]
            print(f"   ...")
            print(f"   Chunk {last_chunk[0]}: {self.format_timestamp(last_chunk[2])}")

        # Process in batches
        all_chunk_analyses = []
        total_batches = (len(file_data) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(0, len(file_data), self.batch_size):
            batch = file_data[batch_idx:batch_idx + self.batch_size]
            current_batch_num = batch_idx // self.batch_size + 1

            print(f"\n📋 Processing batch {current_batch_num}/{total_batches} " +
                  f"(chunks {batch[0][0]}-{batch[-1][0]})")

            batch_analyses = await self.process_transcript_batch(batch, output_folder)
            all_chunk_analyses.extend(batch_analyses)

            print(f"✅ Batch {current_batch_num} complete")

        # Print processing statistics
        elapsed = time.time() - start_time
        print(f"\n📊 Processing Statistics:")
        print(f"  • Total time: {elapsed:.1f}s")
        print(f"  • Chunks processed: {self.processing_stats['chunks_processed']}")
        print(f"  • Chunks cached: {self.processing_stats['chunks_cached']}")
        print(f"  • Average time per chunk: {elapsed / len(transcript_files):.1f}s")

        if self.processing_stats['errors']:
            print(f"  • Errors encountered: {len(self.processing_stats['errors'])}")
            for error in self.processing_stats['errors'][:5]:  # Show first 5 errors
                print(f"    - {error}")

        # Combine all analyses
        print("\n🔄 Combining all chunk analyses...")
        full_analysis = await self.combine_analyses(all_chunk_analyses)

        # Deduplicate and clean up chapters
        full_analysis = self.deduplicate_chapters(full_analysis)

        # Save combined analysis as YAML
        combined_file = output_folder / "full_video_analysis.yaml"
        with open(combined_file, 'w', encoding='utf-8') as f:
            yaml.dump(full_analysis.model_dump(), f,
                      default_flow_style=False,
                      sort_keys=False,
                      allow_unicode=True)
        print(f"💾 Saved combined analysis to {combined_file}")

        # Generate formatted outputs
        await self.generate_formatted_outputs(full_analysis, output_folder)

        return full_analysis

    def deduplicate_chapters(self, analysis: FullVideoAnalysis) -> FullVideoAnalysis:
        """Remove duplicate chapters that might occur due to chunk overlap."""
        if not analysis.chapters:
            return analysis

        cleaned_chapters = []
        last_timestamp = -1

        for chapter in sorted(analysis.chapters, key=lambda x: x.timestamp_seconds):
            # Skip chapters that are too close to the previous one (within overlap window)
            if chapter.timestamp_seconds > last_timestamp + 5:  # 5 second minimum gap
                cleaned_chapters.append(chapter)
                last_timestamp = chapter.timestamp_seconds

        analysis.chapters = cleaned_chapters
        return analysis

    async def combine_analyses(self,
                               chunk_analyses: list[ChunkAnalysis],
                               video_title: str | None = None) -> FullVideoAnalysis:
        """Combine individual chunk analyses into a complete video analysis."""

        # Prepare data for combination
        all_summaries = [analysis.summary for analysis in chunk_analyses]
        all_topics = []
        all_chapters = []
        all_quotes = []
        combined_outline = {}

        for analysis in chunk_analyses:
            all_topics.extend(analysis.key_topics)
            all_chapters.extend(analysis.chapters)
            if analysis.notable_quotes:
                all_quotes.extend(analysis.notable_quotes)

            # Merge outlines
            for topic, subtopics in analysis.topic_outline.items():
                if topic in combined_outline:
                    if isinstance(subtopics, list):
                        combined_outline[topic].extend(
                            [st for st in subtopics if st not in combined_outline[topic]]
                        )
                else:
                    combined_outline[topic] = subtopics if isinstance(subtopics, list) else []

        # Create a context string for the AI
        context = {
            "video_title": video_title or "Video",
            "chunk_summaries": all_summaries,
            "all_topics": list(set(all_topics)),  # Deduplicate
            "raw_outline": combined_outline,
            "all_chapters": [
                {"time": self.format_timestamp(ch.timestamp_seconds),
                 "title": ch.title,
                 "description": ch.description}
                for ch in all_chapters
            ],
            "quotes": all_quotes[:10]  # Limit to top 10 quotes
        }

        system_prompt = """You are creating a comprehensive analysis of a complete video based on individual chunk analyses.

        Synthesize the provided information to create:
        1. An executive summary (2-3 sentences capturing the essence)
        2. A detailed summary (comprehensive but concise overview)
        3. Main topics list (deduplicated and organized by importance)
        4. A complete hierarchical outline (reorganized and cleaned up)
        5. A refined chapter list (combine similar chapters, ensure logical flow)
        6. Key takeaways (5-10 main insights)
        7. Most notable quotes (select the best 3-5 if available)

        For chapters:
        - Combine very similar/redundant chapters
        - Ensure smooth progression through the video
        - Keep timestamps from the original chapters
        - Aim for 10-25 chapters for the full video
        - Remove duplicate chapters that might come from overlapping chunks
        """

        try:
            response = await self.processor.async_make_openai_json_mode_ai_request(
                system_prompt=system_prompt,
                input_data=context,
                output_model=FullVideoAnalysis
            )
            return response

        except Exception as e:
            print(f"Error combining analyses: {e}")
            raise

    async def generate_formatted_outputs(self,
                                         analysis: FullVideoAnalysis,
                                         output_folder: Path):
        """Generate user-friendly formatted outputs."""

        # YouTube description with chapters
        youtube_file = output_folder / "youtube_description.txt"
        with open(youtube_file, 'w', encoding='utf-8') as f:
            f.write("📝 VIDEO SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(analysis.executive_summary + "\n\n")

            f.write("📚 CHAPTERS\n")
            f.write("-" * 50 + "\n")
            for chapter in analysis.chapters:
                timestamp = self.format_timestamp(chapter.timestamp_seconds)
                f.write(f"{timestamp} - {chapter.title}\n")
                if chapter.description:
                    f.write(f"   {chapter.description}\n")

            f.write("\n🎯 KEY TAKEAWAYS\n")
            f.write("-" * 50 + "\n")
            for takeaway in analysis.key_takeaways:
                f.write(f"• {takeaway}\n")

            if analysis.notable_quotes:
                f.write("\n💬 NOTABLE QUOTES\n")
                f.write("-" * 50 + "\n")
                for quote in analysis.notable_quotes:
                    f.write(f'"{quote}"\n\n')

        print(f"📄 Generated YouTube description at {youtube_file}")

        # Detailed markdown report
        markdown_file = output_folder / "video_analysis_report.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write("# Video Analysis Report\n\n")

            f.write("## Executive Summary\n\n")
            f.write(analysis.executive_summary + "\n\n")

            f.write("## Detailed Summary\n\n")
            f.write(analysis.detailed_summary + "\n\n")

            f.write("## Main Topics\n\n")
            for topic in analysis.main_topics:
                f.write(f"- {topic}\n")
            f.write("\n")

            f.write("## Complete Topic Outline\n\n")
            for topic, subtopics in analysis.complete_outline.items():
                f.write(f"### {topic}\n")
                if isinstance(subtopics, list):
                    for subtopic in subtopics:
                        f.write(f"- {subtopic}\n")
                f.write("\n")

            f.write("## Video Chapters\n\n")
            for chapter in analysis.chapters:
                timestamp = self.format_timestamp(chapter.timestamp_seconds)
                f.write(f"**{timestamp}** - {chapter.title}\n")
                if chapter.description:
                    f.write(f"> {chapter.description}\n")
                f.write("\n")

            f.write("## Key Takeaways\n\n")
            for i, takeaway in enumerate(analysis.key_takeaways, 1):
                f.write(f"{i}. {takeaway}\n")

            if analysis.notable_quotes:
                f.write("\n## Notable Quotes\n\n")
                for quote in analysis.notable_quotes:
                    f.write(f"> \"{quote}\"\n\n")

        print(f"📄 Generated markdown report at {markdown_file}")