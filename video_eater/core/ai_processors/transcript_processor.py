import asyncio
import json
import logging
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import yaml
from pydantic import BaseModel, Field

from video_eater.core.ai_processors.base_processor import BaseAIProcessor

logger = logging.getLogger(__name__)


class ChapterHeading(BaseModel):
    timestamp_seconds: float = Field(description="Start time in seconds")
    title: str = Field(description="Chapter title")
    description: str = Field(description="Brief description of what happens in this chapter")


class PullQuote(BaseModel):
    timestamp_seconds: float = Field(description="Start time in seconds when the quote was spoken")
    pull_quotes: list[str] = Field(
        description="Most impactful, interesting, or otherwise notable pull quotes from the video")

    reason_for_selection: str = Field(description="The reason this quote was selected as a pull quote")
    context_around_quote: str = Field(description="Brief context around the quote to explain its significance")


class SubTopicOutlineItem(BaseModel):
    subtopic: str = Field(description="Subtopic under a main topic")
    details: list[str] = Field(description="List of details or points under this subtopic")


class TopicOutlineItem(BaseModel):
    topic: str = Field(description="Main topic")
    topic_overview: str = Field(description="Brief overview of the main topic")
    subtopics: list[SubTopicOutlineItem] = Field(description="List of subtopics under this main topic")


class ChunkAnalysis(BaseModel):
    summary: str = Field(description="Comprehensive summary of the chunk content")
    key_topics: list[str] = Field(description="list of main topics discussed")
    chunk_outline: list[TopicOutlineItem] = Field(description="Hierarchical outline of topics and subtopics")
    chapters: list[ChapterHeading] = Field(description="Timestamped chapter headings")
    pull_quotes: list[PullQuote] = Field(
        description="Most impactful, interesting, or otherwise notable pull quotes from the video")


class FullVideoAnalysis(BaseModel):
    executive_summary: str = Field(description="High-level summary of the entire video")
    detailed_summary: str = Field(description="Comprehensive summary with key points")
    main_themes: list[str] = Field(description="Primary themes covered across the video")
    complete_outline: list[TopicOutlineItem] = Field(description="Hierarchical outline of topics and subtopics")
    chapters: list[ChapterHeading] = Field(description="Full video chapter list with adjusted timestamps")
    key_takeaways: list[str] = Field(description="Main insights and conclusions")
    pull_quotes: list[PullQuote] = Field(
        description="Most impactful, interesting, or otherwise notable pull quotes from the video")


async def _return_cached(idx: int, analysis: ChunkAnalysis):
    """Helper to return cached analysis in async context."""
    return idx, analysis, None


class TranscriptProcessor:
    """Process transcribed chunks to generate summaries, outlines, and chapters."""

    def __init__(self,
                 model: str,
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
                                           chunk_name: str) -> Tuple[int, ChunkAnalysis, str | None]:
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
                raise

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
        3. A hierarchical topic outline using this schema:
           - chunk_outline: list of TopicOutlineItem objects, each with:
             • topic (string)
             • topic_overview (string)
             • subtopics: list of SubTopicOutlineItem objects, each with:
               – subtopic (string)
               – details (list of strings)
        4. Timestamped chapter headings (3-7 chapters, depending on content)
        5. Pull quotes FROM THE transcript_text that either include important insights, clever/interesting/funny turns of phrase, or both. These must be WORD FOR WORD TRANSCRIPTIONS of things that were said in the video/transcripts

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
                        _return_cached(chunk_idx, chunk_analysis)
                    ))
            else:
                # Load transcript
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)

                # Extract text
                if isinstance(transcript_data, dict):
                    transcript_text = transcript_data['full_transcript']
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
                                        chunk_analysis_output_folder: Path) -> FullVideoAnalysis:
        """Process all transcript chunks in a folder with parallel processing."""

        start_time = time.time()

        chunk_analysis_output_folder.mkdir(parents=True, exist_ok=True)

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
        print("📍 Calculated audio/transcript chunk start times:")
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

            print(f"\n📋 Processing Transcript batch {current_batch_num}/{total_batches} " +
                  f"(chunks {batch[0][0]}-{batch[-1][0]})")

            batch_analyses = await self.process_transcript_batch(batch, chunk_analysis_output_folder)
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
        combined_file = chunk_analysis_output_folder.parent / "full_video_analysis.yaml"
        with open(combined_file, 'w', encoding='utf-8') as f:
            yaml.dump(full_analysis.model_dump(), f,
                      default_flow_style=False,
                      sort_keys=False,
                      allow_unicode=True)
        print(f"💾 Saved combined analysis to {combined_file}")

        # Generate formatted outputs
        await self.generate_formatted_outputs(full_analysis, chunk_analysis_output_folder.parent)

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
        # Combined outline structure: {topic: {"topic_overview": str, "subtopics": {subtopic: set(details)}}}

        for analysis in chunk_analyses:
            all_topics.extend(analysis.key_topics)
            all_chapters.extend(analysis.chapters)
            if analysis.pull_quotes:
                all_quotes.extend(analysis.pull_quotes[:3] if len(analysis.pull_quotes) > 2 else analysis.pull_quotes)

            # Merge outlines (TopicOutlineItem -> SubTopicOutlineItem)
            for item in analysis.chunk_outline:
                topic = item.topic
                entry = combined_outline.setdefault(topic, {"topic_overview": item.topic_overview or "", "subtopics": {}})
                if not entry["topic_overview"] and item.topic_overview:
                    entry["topic_overview"] = item.topic_overview
                for st in item.subtopics:
                    st_entry = entry["subtopics"].setdefault(st.subtopic, set())
                    for det in (st.details or []):
                        st_entry.add(det)

        # Create a context string for the AI
        # Build raw outline list for the model input
        raw_outline = []
        for topic, data in combined_outline.items():
            subtopic_list = [
                {"subtopic": st, "details": sorted(list(details))}
                for st, details in data["subtopics"].items()
            ]
            raw_outline.append({
                "topic": topic,
                "topic_overview": data["topic_overview"],
                "subtopics": subtopic_list
            })

        context = {
            "video_title": video_title or "Video",
            "chunk_summaries": all_summaries,
            "all_topics": list(set(all_topics)),  # Deduplicate
            "raw_outline": raw_outline,
            "all_chapters": [
                {"time": self.format_timestamp(ch.timestamp_seconds),
                 "title": ch.title,
                 "description": ch.description}
                for ch in all_chapters
            ],
            "quotes": [pq.model_dump() for pq in all_quotes]
        }

        system_prompt = """You are creating a comprehensive analysis of a complete video based on individual chunk analyses.

        Synthesize the provided information to create the FullVideoAnalysis object:
        1. executive_summary (2-3 sentences capturing the essence)
        2. detailed_summary (comprehensive but concise overview)
        3. main_themes (deduplicated and organized by importance)
        4. complete_outline: list of TopicOutlineItem objects with topic, topic_overview, and subtopics (list of SubTopicOutlineItem with subtopic and details list)
        5. chapters: refined list (combine similar chapters, ensure logical flow)
        6. key_takeaways (5-10 main insights)
        7. pull_quotes: select the best 8-10 PullQuote objects

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

            if analysis.pull_quotes:
                f.write("\n💬 NOTABLE QUOTES\n")
                f.write("-" * 50 + "\n")
                for pq in analysis.pull_quotes:
                    ts = self.format_timestamp(pq.timestamp_seconds)
                    for line in pq.pull_quotes:
                        f.write(f"{ts} - \"{line}\"\n")

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
            for topic in analysis.main_themes:
                f.write(f"- {topic}\n")
            f.write("\n")

            f.write("## Complete Topic Outline\n\n")
            for item in analysis.complete_outline:
                f.write(f"### {item.topic}\n")
                if getattr(item, "topic_overview", None):
                    f.write(f"> {item.topic_overview}\n\n")
                for st in item.subtopics:
                    f.write(f"- {st.subtopic}\n")
                    for det in st.details:
                        f.write(f"  - {det}\n")
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

            if analysis.pull_quotes:
                f.write("\n## Notable Quotes\n\n")
                for pq in analysis.pull_quotes:
                    ts = self.format_timestamp(pq.timestamp_seconds)
                    for line in pq.pull_quotes:
                        f.write(f"> [{ts}] \"{line}\"\n\n")

        print(f"📄 Generated markdown report at {markdown_file}")
