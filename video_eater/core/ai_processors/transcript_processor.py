import asyncio
import json
import yaml
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import timedelta
import re

from video_eater.core.ai_processors.base_processor import BaseAIProcessor
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# Pydantic models for structured outputs
class ChapterHeading(BaseModel):
    timestamp_seconds: float = Field(description="Start time in seconds")
    title: str = Field(description="Chapter title")
    description: str|None = Field(description="Brief description of what happens in this chapter")


class ChunkAnalysis(BaseModel):
    summary: str = Field(description="Comprehensive summary of the chunk content")
    key_topics: list[str] = Field(description="list of main topics discussed")
    topic_outline: dict[str, list[str]|dict[str,object]] = Field(description="Hierarchical outline of topics and subtopics")
    chapters: list[ChapterHeading] = Field(description="Timestamped chapter headings")
    notable_quotes: list[str]|None = Field(default=None, description="Important or interesting quotes")


class FullVideoAnalysis(BaseModel):
    executive_summary: str = Field(description="High-level summary of the entire video")
    detailed_summary: str = Field(description="Comprehensive summary with key points")
    main_topics: list[str] = Field(description="Primary topics covered across the video")
    complete_outline: dict[str, list[str]] = Field(description="Complete hierarchical topic outline")
    chapters: list[ChapterHeading] = Field(description="Full video chapter list with adjusted timestamps")
    key_takeaways: list[str] = Field(description="Main insights and conclusions")
    notable_quotes: list[str]|None = Field(default=None, description="Most impactful quotes from the video")


class TranscriptProcessor:
    """Process transcribed chunks to generate summaries, outlines, and chapters."""

    def __init__(self, model: str = "deepseek-chat", use_async: bool = True):
        self.processor = BaseAIProcessor(model=model, use_async=use_async)
        self.model = model

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

    @staticmethod
    def parse_chunk_filename(filename: str) -> float:
        """Extract the start time in seconds from chunk filename."""
        # Pattern: chunk_XXX_HHh-MMm-SSsec
        pattern = r"chunk_\d+_(\d+)h-(\d+)m-(\d+)sec"
        match = re.search(pattern, filename)
        if match:
            hours, minutes, seconds = map(int, match.groups())
            return hours * 3600 + minutes * 60 + seconds
        return 0.0

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

    async def combine_analyses(self,
                               chunk_analyses: list[ChunkAnalysis],
                               video_title: str|None = None) -> FullVideoAnalysis:
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
                    # Merge subtopics, avoiding duplicates
                    combined_outline[topic].extend(
                        [st for st in subtopics if st not in combined_outline[topic]]
                    )
                else:
                    combined_outline[topic] = subtopics

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
        - Aim for 10-20 chapters for the full video
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

    async def process_transcript_folder(self,
                                        transcript_folder: Path,
                                        output_folder: Path|None = None) -> FullVideoAnalysis:
        """Process all transcript chunks in a folder."""

        # Set up output folder
        if output_folder is None:
            output_folder = transcript_folder.parent / "analysis"
        output_folder.mkdir(parents=True, exist_ok=True)

        # Find all transcript JSON files
        transcript_files = sorted(transcript_folder.glob("*.transcript.json"))
        if not transcript_files:
            raise ValueError(f"No transcript files found in {transcript_folder}")

        print(f"Found {len(transcript_files)} transcript files to process")

        # Process each chunk
        chunk_analyses = []
        for idx, transcript_file in enumerate(transcript_files):
            print(f"Processing chunk {idx + 1}/{len(transcript_files)}: {transcript_file.name}")

            # Check if analysis already exists (now as YAML)
            analysis_file = output_folder / f"{transcript_file.stem}.analysis.yaml"

            if analysis_file.exists():
                print(f"Loading existing analysis from {analysis_file}")
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    chunk_analysis = ChunkAnalysis(**yaml.safe_load(f))
            else:
                # Load transcript (still JSON from Whisper API)
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)

                # Extract text (handling different possible structures)
                if isinstance(transcript_data, dict):
                    transcript_text = transcript_data.get('text', '')
                else:
                    transcript_text = str(transcript_data)

                # Get chunk start time from filename
                chunk_start = self.parse_chunk_filename(transcript_file.name)

                # Analyze chunk
                chunk_analysis = await self.analyze_chunk(
                    transcript_text=transcript_text,
                    chunk_start_seconds=chunk_start,
                    chunk_index=idx
                )

                # Save chunk analysis as YAML
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    yaml.dump(chunk_analysis.model_dump(), f,
                              default_flow_style=False,
                              sort_keys=False,
                              allow_unicode=True)
                print(f"Saved chunk analysis to {analysis_file}")

            chunk_analyses.append(chunk_analysis)

        # Combine all analyses
        print("Combining all chunk analyses...")
        full_analysis = await self.combine_analyses(chunk_analyses)

        # Save combined analysis as YAML
        combined_file = output_folder / "full_video_analysis.yaml"
        with open(combined_file, 'w', encoding='utf-8') as f:
            yaml.dump(full_analysis.model_dump(), f,
                      default_flow_style=False,
                      sort_keys=False,
                      allow_unicode=True)
        print(f"Saved combined analysis to {combined_file}")

        # Generate formatted outputs
        await self.generate_formatted_outputs(full_analysis, output_folder)

        return full_analysis

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

        print(f"Generated YouTube description at {youtube_file}")

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

        print(f"Generated markdown report at {markdown_file}")


# Example usage
async def main():
    # Example: Process a folder of transcripts
    transcript_folder = Path(r"\\jon-nas\jon-nas\videos\livestream_videos\2025-08-14-JSM-Livestream-Skellycam\chunk_transcripts")

    processor = TranscriptProcessor()
    full_analysis = await processor.process_transcript_folder(transcript_folder)

    print("Analysis complete!")
    print(f"Executive Summary: {full_analysis.executive_summary}")
    print(f"Found {len(full_analysis.chapters)} chapters")
    print(f"Main topics: {', '.join(full_analysis.main_topics[:5])}")


if __name__ == "__main__":
    asyncio.run(main())