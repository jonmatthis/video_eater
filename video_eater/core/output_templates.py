# output_templates.py (Updated YouTubeDescriptionFormatter)
from typing import Protocol

from jinja2 import Template

from video_eater.core.ai_processors.ai_prompt_models import PromptModel, FullVideoAnalysis


class OutputFormatter(Protocol):
    """Protocol for output formatters."""

    def format(self, analysis: PromptModel) -> str:
        ...


AI_DISCLAIMER_AND_SOURCE_CODE = ("```\n"
                                 "AI generated summary - anticipate wonk.\n"
                                 "Generated via: https://github.com/jonmatthis/video_eater\n"
                                 "```\n\n")


class YouTubeDescriptionFormatter:
    """Format analysis for YouTube descriptions."""

    template = Template("""
{{ ai_disclaimer_and_source_code }}

📝 VIDEO SUMMARY
{{ '-' * 50 }}
{{ analysis.summary.one_sentence_summary }}


📚 CHAPTERS
{{ '-' * 50 }}
{% for i, chunk in visible_chapters %}
{{ chunk.as_chapter_heading_with_description if include_chapter_descriptions[i] else chunk.as_chapter_heading_without_description }}
{% endfor %}

{% if pull_quotes_to_show %}
💬 PULL QUOTES
{{ '-' * 50 }}
{% for quote in pull_quotes_to_show %}
• {{ quote.as_string_youtube_formatted_timestamp }}
{% endfor %}
{% endif %}

🎯 KEY TAKEAWAYS
{{ '-' * 50 }}
{% for takeaway in analysis.takeaways %}
• {{ takeaway }}
{% endfor %}

🤔 TOPICS COVERED
{{ '-' * 50 }}
{% for topic in analysis.topics %}
• {{ topic }}

{% endfor %}

💭 MAIN THEMES
{{ '-' * 50 }}
{% for theme in analysis.themes %}
• {{ theme }}
{% endfor %}
""")

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to YouTube timestamp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def format(self, analysis: FullVideoAnalysis, max_length: int = 5000) -> str:
        """
        Format the analysis with sophisticated length trimming.

        Args:
            analysis: The full video analysis to format
            max_length: Maximum character length for the output

        Returns:
            Formatted string within the length constraints
        """
        # Initialize with most verbose settings
        include_chapter_descriptions = [True] * len(analysis.chunk_analyses)
        visible_chapter_indices = list(range(len(analysis.chunk_analyses)))
        max_pull_quotes = min(20, len(analysis.pull_quotes))  # Start with max 20

        # Track what we've tried to trim
        trimming_stage = 0
        max_trimming_stages = 1000  # Safety to prevent infinite loops
        rendered_string = ""

        while trimming_stage < max_trimming_stages:
            rendered_string = self._render_with_settings(
                analysis=analysis,
                include_chapter_descriptions=include_chapter_descriptions,
                visible_chapter_indices=visible_chapter_indices,
                max_pull_quotes=max_pull_quotes
            )
            rendered_length = len(rendered_string)

            if rendered_length <= max_length:
                print(f"Final formatted string length: {rendered_length} characters")
                return rendered_string

            print(f"Trimming stage {trimming_stage}: Current length {rendered_length}, target {max_length}")

            # Stage 1: Trim themes, topics, takeaways using the model's trim method
            if self._can_trim_analysis_lists(analysis=analysis):
                trimmed = analysis.trim()
                if trimmed:
                    print(f"  Trimmed analysis lists (themes/topics/takeaways)")
                else:
                    print(f"  Could not trim analysis lists further")

            # Stage 2: Reduce pull quotes from 20 to 10
            elif max_pull_quotes > 10:
                max_pull_quotes = 10
                print(f"  Reduced pull quotes to {max_pull_quotes}")

            # Stage 3: Start removing chapter descriptions from the end
            elif any(include_chapter_descriptions[i] for i in visible_chapter_indices):
                # Find the last visible chapter with description still included
                for i in reversed(visible_chapter_indices):
                    if include_chapter_descriptions[i]:
                        include_chapter_descriptions[i] = False
                        print(f"  Removed description for chapter {i}")
                        break

            # Stage 4: Reduce pull quotes from 10 to 3
            elif max_pull_quotes > 3:
                max_pull_quotes = 3
                print(f"  Reduced pull quotes to {max_pull_quotes}")

            # Stage 5: Start removing chapters using the specific pattern
            elif len(visible_chapter_indices) > 1:
                visible_chapter_indices = self._remove_chapter_by_pattern(
                    visible_chapter_indices=visible_chapter_indices)
                print(f"  Removed chapter, now showing {len(visible_chapter_indices)} chapters")

            # Stage 6: Further reduce pull quotes if needed
            elif max_pull_quotes > 0:
                max_pull_quotes = max(0, max_pull_quotes - 1)
                print(f"  Reduced pull quotes to {max_pull_quotes}")

            # Final stage: Everything has been trimmed
            else:
                print(f"WARNING: Unable to format within {max_length} characters after all trimming attempts")
                break

            trimming_stage += 1

        # Fallback if we hit max stages
        print(f"WARNING: Hit maximum trimming stages. Returning at length {len(rendered_string)} characters")
        return rendered_string

    def _remove_chapter_by_pattern(self, visible_chapter_indices: list[int]) -> list[int]:
        """
        Remove chapters following the pattern: second from end, then by 2's from there.

        Args:
            visible_chapter_indices: Current list of visible chapter indices

        Returns:
            Updated list with one chapter removed
        """
        if len(visible_chapter_indices) <= 1:
            return visible_chapter_indices

        # For the first removal, remove second from end
        if len(visible_chapter_indices) == len(range(max(visible_chapter_indices) + 1)):
            # This is the first removal
            if len(visible_chapter_indices) >= 2:
                visible_chapter_indices.pop(-2)
        else:
            # Subsequent removals: remove every other chapter starting from second position
            # Find which indices to keep (every other one, keeping first and last if possible)
            new_indices = []
            for i, idx in enumerate(visible_chapter_indices):
                # Keep first, and then every other one
                if i == 0 or i == len(visible_chapter_indices) - 1:
                    new_indices.append(idx)
                elif i % 2 == 0:
                    new_indices.append(idx)

            # If we didn't remove anything with the above logic, remove from middle
            if len(new_indices) == len(visible_chapter_indices) and len(visible_chapter_indices) > 2:
                mid = len(visible_chapter_indices) // 2
                visible_chapter_indices.pop(mid)
            else:
                visible_chapter_indices = new_indices

        return visible_chapter_indices

    def _can_trim_analysis_lists(self, analysis: FullVideoAnalysis) -> bool:
        """
        Check if we can still trim themes, topics, or takeaways.

        Args:
            analysis: The full video analysis

        Returns:
            True if any list can still be trimmed
        """
        # Check if any of the lists are above their minimum thresholds
        # Note: pull_quotes trimming is now handled separately in the main format method
        return (
                len(analysis.themes) > 3 or
                len(analysis.topics) > 3 or
                len(analysis.takeaways) > 3
        )

    def _render_with_settings(
            self,
            analysis: FullVideoAnalysis,
            include_chapter_descriptions: list[bool],
            visible_chapter_indices: list[int],
            max_pull_quotes: int
    ) -> str:
        """
        Render the template with specific settings.

        Args:
            analysis: The full video analysis
            include_chapter_descriptions: List of booleans for each chapter
            visible_chapter_indices: List of indices for chapters to show
            max_pull_quotes: Maximum number of pull quotes to show

        Returns:
            Rendered string
        """
        # Prepare visible chapters with their original indices
        visible_chapters = [(i, analysis.chunk_analyses[i]) for i in visible_chapter_indices]

        # Get the pull quotes to show (top N by quality)
        pull_quotes_to_show = analysis.get_pull_quotes(sort_by="quality")[:max_pull_quotes]

        sorted_pull_quotes_to_show = sorted(pull_quotes_to_show, key=lambda pq: pq.timestamp_seconds)
        rendered = self.template.render(
            analysis=analysis,
            format_timestamp=self.format_timestamp,
            include_chapter_descriptions=include_chapter_descriptions,
            visible_chapters=visible_chapters,
            pull_quotes_to_show=sorted_pull_quotes_to_show,
            ai_disclaimer_and_source_code=AI_DISCLAIMER_AND_SOURCE_CODE
        )
        # Clean up excessive newlines
        return rendered.replace("\n\n\n", "\n\n")


class MarkdownReportFormatter:
    """Format analysis as detailed markdown report."""

    def format(self, analysis: FullVideoAnalysis) -> str:
        return analysis.to_markdown(disclaimer_text=AI_DISCLAIMER_AND_SOURCE_CODE)


class JsonFormatter:
    """Format analysis as JSON for programmatic use."""

    def format(self, analysis: FullVideoAnalysis) -> str:
        import json
        return json.dumps(analysis.model_dump(), indent=2, ensure_ascii=False)


class SimpleTextFormatter:
    """Format analysis as simple text summary."""

    template = Template("""{{ analysis.summary }}""")

    def format(self, analysis: FullVideoAnalysis) -> str:
        return self.template.render(analysis=analysis)