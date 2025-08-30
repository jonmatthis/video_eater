# output_templates.py
from typing import Protocol

from jinja2 import Template

from video_eater.core.ai_processors.ai_prompt_models import PromptModel, FullVideoAnalysis


class OutputFormatter(Protocol):
    """Protocol for output formatters."""

    def format(self, analysis: PromptModel) -> str:
        ...

AI_DISCLAIMER_AND_SOURCE_CODE = "(AI generated summary - anticipate wonk. Generated code available here: https://github.com/jonmatthis/video_eater)"

class YouTubeDescriptionFormatter:
    """Format analysis for YouTube descriptions."""

    template = Template("""
    
📝 VIDEO SUMMARY
{{ analysis.executive_summary }}

📚 CHAPTERS
{{ '-' * 50 }}
{% for chunk in analysis.chunk_analyses %}
{{ chunk.as_chapter_heading }}
{% endfor %}

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

{% if analysis.pull_quotes %}
💬 NOTABLE QUOTES
{{ '-' * 50 }}
{% for quote in analysis.pull_quotes %}
{{ quote.as_string_youtube_formatted_timestamp }}"
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


    def format(self, analysis:FullVideoAnalysis, max_length:int=5000) -> str:
        got_valid_string = False
        rendered_string = ""
        while not got_valid_string:
            rendered_string =  self.template.render(
                analysis=analysis,
                format_timestamp=self.format_timestamp
            )
            if len(rendered_string) <= max_length:
                got_valid_string = True
            else:
                analysis.trim()

        return AI_DISCLAIMER_AND_SOURCE_CODE + "\n\n" +  rendered_string


class MarkdownReportFormatter:
    """Format analysis as detailed markdown report."""

    def format(self, analysis:FullVideoAnalysis) -> str:
        return str(analysis.to_markdown(disclaimer_text=AI_DISCLAIMER_AND_SOURCE_CODE))


class JsonFormatter:
    """Format analysis as JSON for programmatic use."""

    def format(self, analysis) -> str:
        import json
        return json.dumps(analysis.model_dump(), indent=2, ensure_ascii=False)


class SimpleTextFormatter:
    """Format analysis as simple text summary."""

    template = Template("""{{ analysis.summary }}""")

    def format(self, analysis:FullVideoAnalysis) -> str:
        return self.template.render(analysis=analysis)


