# output_templates.py
from typing import Protocol

from jinja2 import Template

from video_eater.core.ai_processors.ai_prompt_models import PromptModel, FullVideoAnalysis


class OutputFormatter(Protocol):
    """Protocol for output formatters."""

    def format(self, analysis: PromptModel) -> str:
        ...


class YouTubeDescriptionFormatter:
    """Format analysis for YouTube descriptions."""

    template = Template("""📝 VIDEO SUMMARY
(AI generated summary - expect wonkiness. Generated code available here: https://github.com/jonmatthis/video_eater)

{{ analysis.detailed_summary }}

📚 CHAPTERS
{{ '-' * 50 }}
{% for chapter in analysis.chapters %}
{{ format_timestamp(chapter.chapter_start_timestamp_seconds) }} - {{ chapter.title }}
{% if chapter.description %}   {{ chapter.description }}{% endif %}
{% endfor %}

🎯 KEY TAKEAWAYS
{{ '-' * 50 }}
{% for takeaway in analysis.key_takeaways %}
• {{ takeaway }}
{% endfor %}

{% if analysis.pull_quotes %}
💬 NOTABLE QUOTES
{{ '-' * 50 }}
{% for quote in analysis.pull_quotes %}
{{ format_timestamp(quote.timestamp_seconds) }} - "{{ quote.text_content }}"
{% endfor %}
{% endif %}
""")

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to YouTube timestamp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


    def format(self, analysis:FullVideoAnalysis) -> str:
        return self.template.render(
            analysis=analysis,
            format_timestamp=self.format_timestamp
        )


class MarkdownReportFormatter:
    """Format analysis as detailed markdown report."""

    template = Template("""# Video Analysis Report
## Executive Summary
{{ analysis.executive_summary }}
## Detailed Summary
{{ analysis.detailed_summary }}
## Main Topics
{% for topic in analysis.main_themes %}
- {{ topic }}
{% endfor %}
## Complete Topic Outline
{% for item in analysis.complete_outline %}
### {{ item.topic }}
{% if item.topic_overview %}
> {{ item.topic_overview }}
{% endif %}
{% for subtopic in item.subtopics %}
- {{ subtopic.subtopic }}
{% for detail in subtopic.details %}
  - {{ detail }}
{% endfor %}
{% endfor %}
{% endfor %}
## Video Chapters
{% for chapter in analysis.chapters %}
**{{ format_timestamp(chapter.chapter_start_timestamp_seconds) }}** - {{ chapter.title }}
{% if chapter.description %}
> {{ chapter.description }}
{% endif %}
{% endfor %}
## Key Takeaways
{% for takeaway in analysis.key_takeaways %}
{{ loop.index }}. {{ takeaway }}
{% endfor %}
{% if analysis.pull_quotes %}
## Notable Quotes
{% for quote in analysis.pull_quotes %}
> [{{ format_timestamp(quote.timestamp_seconds) }}] "{{ quote.text_content }}"

- Reason for selection:

    {{ quote.reason_for_selection }}
    
- Context around Quote:

    {{ quote.context_around_quote }}
    
{% endfor %}
{% endif %}
{% if analysis.particularly_interesting_60_second_clips %}
## Particularly Interesting 60-Second Clips
{% for clip in analysis.particularly_interesting_60_second_clips %}
### Clip from {{ format_timestamp(clip.start_timestamp_seconds) }} to {{ format_timestamp(clip.end_timestamp_seconds) }}
> {{ clip.text_content }}

- **Reason for selection:**

    {{ clip.reason_for_selection }}
    
- **Context around Clip:**

    {{ clip.context_around_clip }}
    
{% endfor %}
{% endif %}
""")

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to timestamp format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def format(self, analysis) -> str:
        return self.template.render(
            analysis=analysis,
            format_timestamp=self.format_timestamp
        )


class JsonFormatter:
    """Format analysis as JSON for programmatic use."""

    def format(self, analysis) -> str:
        import json
        return json.dumps(analysis.model_dump(), indent=2, ensure_ascii=False)


class SimpleTextFormatter:
    """Format analysis as simple text summary."""

    template = Template("""{{ analysis.executive_summary }}

Main Topics:
{% for topic in analysis.main_themes[:5] %}
- {{ topic }}
{% endfor %}

Key Takeaways:
{% for takeaway in analysis.key_takeaways[:5] %}
- {{ takeaway }}
{% endfor %}
""")

    def format(self, analysis) -> str:
        return self.template.render(analysis=analysis)


