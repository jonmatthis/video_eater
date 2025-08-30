import asyncio
import json
import logging
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Tuple

import yaml

from video_eater.core.ai_processors.ai_prompt_models import ChunkAnalysis, FullVideoAnalysis, SummaryGeneration,  ClipSelection
from video_eater.core.ai_processors.base_processor import BaseAIProcessor
from video_eater.core.output_templates import YouTubeDescriptionFormatter, JsonFormatter, MarkdownReportFormatter, \
    SimpleTextFormatter
from video_eater.core.transcribe_audio.transcript_models import VideoTranscript

logger = logging.getLogger(__name__)


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
        """Convert seconds to YouTube-style timestamp (HH:MM:SS)."""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def parse_chunk_filename(filename: str) -> Tuple[int, float]:
        """Extract chunk index and start time from chunk filename.
        Returns: (chunk_index, start_time_seconds)
        """
        # Match the actual format: *_chunk_XXX__YYY.Ysec* where XXX is the index and YYY.Y is the start time
        pattern = r"chunk_(\d{3})__(\d+(?:\.\d+)?)sec"
        match = re.search(pattern, filename)
        if match:
            chunk_index = int(match.group(1))
            start_time = float(match.group(2))
            return chunk_index, start_time

        # Fallback: try to get just the chunk index
        index_pattern = r"chunk_(\d+)"
        index_match = re.search(index_pattern, filename)
        if index_match:
            return int(index_match.group(1)), 0.0

        return 0, 0.0

    async def analyze_chunk_with_semaphore(self,
                                           transcript_data: VideoTranscript,
                                           chunk_index: int,
                                           chunk_name: str) -> Tuple[int, ChunkAnalysis, str | None]:
        """Analyze a single chunk with rate limiting via semaphore."""
        async with self._semaphore:
            try:
                start_time = time.time()
                print(
                    f"  📝 Processing chunk {chunk_index + 1}: {chunk_name} (starts at {self.format_timestamp(transcript_data.start_time)})")

                analysis = await self.analyze_chunk(
                    transcript_data=transcript_data,
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
                            transcript_data: VideoTranscript,
                            chunk_index: int = 0) -> ChunkAnalysis:
        """Analyze a single transcript chunk."""

        system_prompt = f"""You are analyzing a transcript chunk from a longer video. 
        This is chunk #{chunk_index} starting at {self.format_timestamp(transcript_data.start_time)} in the video.

       <<<Transcript-START>>>
        {transcript_data.full_transcript}
        <<<Transcript-END>>>
        
        Use this information and provie your answer in JSON format according to the provided schema. 
        
        You must use the information from the transcript to fill in the fields as accurately as possible, in effect to best
        summarize and outline the content of this chunk of the video. Ensure precise and careful copying of the direct quotes.
        """

        try:
            response = await self.processor.async_make_openai_json_mode_ai_request(
                system_prompt=system_prompt,
                input_data={},
                output_model=ChunkAnalysis
            )

            return response

        except Exception as e:
            print(f"Error analyzing chunk {chunk_index}: {e}")
            raise

    async def process_transcript_batch(self,
                                       batch: List[Tuple[int, Path]],
                                       output_folder: Path) -> List[ChunkAnalysis]:
        """Process a batch of transcript files in parallel."""
        tasks = []

        for batch_idx, (chunk_idx, transcript_file) in enumerate(batch):
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
                    transcript_data = VideoTranscript(**json.load(f))



                # Create analysis task
                task = asyncio.create_task(
                    self.analyze_chunk_with_semaphore(
                        transcript_data=transcript_data,
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
                                        chunk_analysis_output_folder: Path,
                                        full_output_folder:Path) -> FullVideoAnalysis:
        """Process all transcript chunks in a folder with parallel processing."""

        start_time = time.perf_counter()

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
            file_data.append((chunk_index, transcript_file))

        # Sort by chunk index to ensure proper ordering
        file_data.sort(key=lambda x: x[0])


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
        elapsed = time.perf_counter() - start_time
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

        # # Generate formatted outputs
        # await self.generate_formatted_outputs(full_analysis,
        #                                         output_folder=full_output_folder)

        return full_analysis

    def deduplicate_chapters(self, analysis: FullVideoAnalysis) -> FullVideoAnalysis:
        """Remove duplicate chapters that might occur due to chunk overlap."""
        if not analysis.chapters:
            return analysis

        cleaned_chapters = []
        last_timestamp = -1

        for chapter in sorted(analysis.chapters, key=lambda x: x.chapter_start_timestamp_seconds):
            # Skip chapters that are too close to the previous one (within overlap window)
            if chapter.chapter_start_timestamp_seconds > last_timestamp + 5:  # 5 second minimum gap
                cleaned_chapters.append(chapter)
                last_timestamp = chapter.chapter_start_timestamp_seconds

        analysis.chapters = cleaned_chapters
        return analysis

    async def combine_analyses(self,
                               chunk_analyses: list[ChunkAnalysis]) -> FullVideoAnalysis:
        pass
        """Combine individual chunk analyses into a complete video analysis."""

        # Stage 1: Aggregate all data from chunks
        all_summaries = []
        all_topics = []
        all_chapters = []
        all_pull_quotes = []
        all_clips = []
        all_outlines = []

        for chunk in chunk_analyses:
            all_summaries.append(chunk.summary)
            all_topics.extend(chunk.key_topics)
            all_chapters.extend(chunk.chapters)
            all_pull_quotes.extend(chunk.pull_quotes)
            if chunk.most_interesting_short_section:
                all_clips.append(chunk.most_interesting_short_section)
            all_outlines.extend(chunk.chunk_outline)

        # Stage 2: Generate executive and detailed summaries first
        summary_prompt = f"""Based on these chunk summaries from a video, create:
        Summaries:
        1. Executive Summary: A concise 3-5 sentence high-level summary capturing the essence of the entire video.
        2. Detailed Chronological Summary: A comprehensive summary broken down chronologically, covering key points in the order they were presented in the video.
        3. Chronological Outline: A detailed outline broken down chronologically, covering key points in the order they were presented in the video and outline structure splitting actions and sub-actions and topics and sub-topics.

        Chunk Summaries:
        {chr(10).join([f"<<<START Chunk {i + 1}>>>>: {summary}  <<<END Chunk {i + 1}>>>>|||||||||||||||||" for i, summary in enumerate(all_summaries)])}

        Provide your response in JSON format in accordance to the provided schema
        """

        summary_response = await self.processor.async_make_openai_json_mode_ai_request(
            system_prompt=summary_prompt,
            input_data={},
            output_model=SummaryGeneration
        )







        # Stage 5: Extract main themes and key takeaways
        themes_prompt = f"""Based on this video analysis, identify:
        1. main_themes: 5-8 primary themes (deduplicated and prioritized)
        2. key_takeaways: 5-10 main insights and conclusions

        Executive Summary: {summary_response.executive_summary}

        All Topics from chunks:
        {chr(10).join(list(set(all_topics))[:50])}  # Dedupe and limit for context

        Video Outline Main Topics:
        {chr(10).join([f"- {topic.topic}: {topic.topic_overview}" for topic in outline_response.complete_outline])}

        respond in JSON format with keys according to the provided schema.
        """

        themes_response = await self.processor.async_make_openai_json_mode_ai_request(
            system_prompt=themes_prompt,
            input_data={},
            output_model=ThemesAndTakeaways
        )


        # Stage 6: Rank and select top pull quotes
        if len(all_pull_quotes) > 10:
            quotes_prompt = f"""Rank these pull quotes and select the TOP 10 most impactful ones.

            Video Summary: {summary_response.executive_summary}

            All Pull Quotes:
            {[q.model_dump_json(indent=2) for i, q in enumerate(all_pull_quotes)]}

            Select quotes that are:
            - Most insightful or thought-provoking
            - Representative of main themes
            - Memorable and quotable
            - Diverse (covering different parts/topics)

            Respond in JSON Format according to the provided schema
            """

            quotes_response = await self.processor.async_make_openai_json_mode_ai_request(
                system_prompt=quotes_prompt,
                input_data={},
                output_model=QuoteSelection
            )

            top_pull_quotes = quotes_response.top_pull_quotes
        else:
            top_pull_quotes = all_pull_quotes
        top_pull_quotes = sorted(top_pull_quotes, key=lambda x: x.timestamp_seconds)
        # remove duplicates based on text_content
        seen_quotes = set()
        unique_pull_quotes = []
        for quote in top_pull_quotes:
            if quote.text_content not in seen_quotes:
                unique_pull_quotes.append(quote)
                seen_quotes.add(quote.text_content)
        top_pull_quotes = unique_pull_quotes
        # Stage 7: Rank and select top 60-second clips
        if len(all_clips) > 10:
            clips_prompt = f"""Rank these 60-second clips and select the TOP 10 most interesting ones.

            Video Summary: {summary_response.executive_summary}

            All Clips:
            {[c.model_dump_json(indent=2) for c in all_clips]}

            Select clips that would work best as standalone content for social media (TikTok, YouTube Shorts, etc):
            - Self-contained stories or insights
            - High entertainment or educational value
            - Strong hooks and conclusions
            - Diverse topics and moments

            Respond in JSON Format according to the provided schema
                        """

            clips_response = await self.processor.async_make_openai_json_mode_ai_request(
                system_prompt=clips_prompt,
                input_data={},
                output_model=ClipSelection
            )

            top_clips = clips_response.top_clips
        else:
            top_clips = all_clips

        # Create final FullVideoAnalysis
        full_analysis = FullVideoAnalysis(
            executive_summary=summary_response.executive_summary,
            detailed_summary=summary_response.detailed_summary,
            main_themes=themes_response.main_themes,
            complete_outline=outline_response.complete_outline,
            chapters=chapters_response.chapters,
            key_takeaways=themes_response.key_takeaways,
            pull_quotes=top_pull_quotes,
            particularly_interesting_60_second_clips=top_clips
        )

        return full_analysis

    async def generate_formatted_outputs(self,
                                         analysis: FullVideoAnalysis,
                                         output_folder: Path,):
        """Generate user-friendly formatted outputs using template formatters."""
        output_folder.mkdir(parents=True, exist_ok=True)
        # Initialize formatters
        youtube_formatter = YouTubeDescriptionFormatter()
        markdown_formatter = MarkdownReportFormatter()
        json_formatter = JsonFormatter()
        simple_formatter = SimpleTextFormatter()

        # YouTube description with chapters
        youtube_file = output_folder / "youtube_description.txt"
        youtube_content = youtube_formatter.format(analysis)
        with open(youtube_file, 'w', encoding='utf-8') as f:
            f.write(youtube_content)
        print(f"📄 Generated YouTube description at {youtube_file}")

        # Detailed markdown report
        markdown_file = output_folder / "video_analysis_report.md"
        markdown_content = markdown_formatter.format(analysis)
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"📄 Generated markdown report at {markdown_file}")

        # JSON output for programmatic use
        json_file = output_folder / "video_analysis.json"
        json_content = json_formatter.format(analysis)
        with open(json_file, 'w', encoding='utf-8') as f:
            f.write(json_content)
        print(f"📄 Generated JSON output at {json_file}")

        # Simple text summary
        summary_file = output_folder / "video_summary.txt"
        summary_content = simple_formatter.format(analysis)
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        print(f"📄 Generated simple summary at {summary_file}")