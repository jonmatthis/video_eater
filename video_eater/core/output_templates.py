# output_templates.py
from typing import Protocol
from pathlib import Path
import yaml
from jinja2 import Template


class OutputFormatter(Protocol):
    """Protocol for output formatters."""

    def format(self, analysis: 'FullVideoAnalysis') -> str:
        ...


class YouTubeDescriptionFormatter:
    """Format analysis for YouTube descriptions."""

    template = Template("""
VIDEO SUMMARY
{{ '=' * 50 }}

{{ analysis.executive_summary }}

CHAPTERS
{{ '-' * 50 }}
{% for chapter in analysis.chapters %}
{{ format_timestamp(chapter.timestamp_seconds) }} - {{ chapter.title }}
{% if chapter.description %}   {{ chapter.description }}{% endif %}
{% endfor %}

KEY TAKEAWAYS
{{ '-' * 50 }}
{% for takeaway in analysis.key_takeaways %}
- {{ takeaway }}
{% endfor %}

{% if analysis.pull_quotes %}
NOTABLE QUOTES
{{ '-' * 50 }}
{% for quote in analysis.pull_quotes %}
{{ format_timestamp(quote.timestamp_seconds) }} - "{{ quote.pull_quotes[0] }}"
{% endfor %}
{% endif %}
""")

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to YouTube timestamp."""
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


class MarkdownReportFormatter:
    """Format analysis as detailed markdown report."""

    template_path = Path("templates/report.md.jinja2")

    def format(self, analysis) -> str:
        # Load template from file for easier editing
        with open(self.template_path) as f:
            template = Template(f.read())

        return template.render(analysis=analysis)