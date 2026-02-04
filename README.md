# video_eater

# AI GENERATED README - ANTICIPATE WONK

A pipeline for extracting audio from videos, transcribing it, and generating AI-powered analysis outputs.

## Disclaimer

This is a personal tool I built for my own use. It is not polished, not well-tested outside my specific environment, and may not work for you. The code is messy in places. Use at your own risk. No support is provided.

## What it does

1. Accepts local video files or YouTube URLs (including playlists)
2. Extracts audio from videos using ffmpeg
3. Chunks audio into overlapping segments (default: 10 minute chunks with 15 second overlap)
4. Transcribes audio chunks using one of: OpenAI Whisper API, AssemblyAI, or local Whisper
5. Analyzes transcripts using an LLM (OpenAI or Deepseek) to extract summaries, themes, topics, pull quotes, etc.
6. Generates output files in multiple formats

## Requirements

- Python 3.10+
- ffmpeg (must be in PATH)
- API keys for whichever services you use:
  - `OPENAI_API_KEY` (required for transcription and/or analysis)
  - `ASSEMBLY_AI_API_KEY` (if using AssemblyAI transcription)
  - `DEEPSEEK_API_KEY` (if using Deepseek for analysis)

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd video_eater

# Install with uv
uv pip install -e .

# Or with pip
pip install -e .
```

Set environment variables (or use a `.env` file in the project root, see `sample.env` for example):

```bash
export OPENAI_API_KEY="sk-..."
export ASSEMBLY_AI_API_KEY="..."  # optional
export DEEPSEEK_API_KEY="..."     # optional
```

## Usage

### Basic usage

Edit the `DEFAULT_VIDEO_INPUTS` list in `__main__.py` to include your video paths or YouTube URLs (videos or playlists should both work) , then run:

```bash
python -m video_eater
```

### Programmatic usage

```python
import asyncio
from pathlib import Path
from video_eater.core.config_models import ProcessingConfig, VideoProject
from video_eater.core.pipeline import VideoProcessingPipeline

config = ProcessingConfig()
pipeline = VideoProcessingPipeline(config=config)

project = VideoProject(video_path=Path("/path/to/video.mp4"))
result = asyncio.run(pipeline.process_video(project=project))
print(result.summary_report())
```

### Configuration options

`ProcessingConfig` accepts the following parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_length_seconds` | 600 | Length of each audio chunk in seconds |
| `chunk_overlap_seconds` | 15 | Overlap between consecutive chunks |
| `analysis_model` | `"gpt-4.1-nano"` | LLM model for analysis |
| `transcription_provider` | `"assembly_ai"` | One of: `"openai"`, `"assembly_ai"`, `"local_whisper"` |
| `whisper_model` | `"large"` | Whisper model size (for local transcription) |
| `max_concurrent_chunks` | 50 | Max concurrent API calls |
| `batch_size` | 10 | Batch size for processing |
| `force_chunk_audio` | False | Re-extract audio even if chunks exist |
| `force_transcribe` | False | Re-transcribe even if transcripts exist |
| `force_analyze` | False | Re-analyze even if analysis exists |

## How to know if it ran successfully

If the pipeline completes without errors, you will see:

1. A summary report printed to stdout showing processing statistics
2. Output files created in the output directory (see below)

The output directory location is printed at the end of the summary report.

If something fails, the pipeline throws exceptions. Check stderr for error messages.

## Output directory structure

For a video at `/path/to/video.mp4`, the following directory structure is created:

```
/path/to/
├── video.mp4
├── video.wav                          # Extracted audio (full)
├── chunks/
│   ├── audio_chunks/
│   │   ├── video_chunk_000__0.0sec.mp3
│   │   ├── video_chunk_001__585.0sec.mp3
│   │   └── ...
│   ├── transcript_chunks/
│   │   ├── video_chunk_000__0.0sec.transcript.json
│   │   ├── video_chunk_001__585.0sec.transcript.json
│   │   └── ...
│   └── analysis_chunks/
│       ├── video_chunk_000__0.0sec.analysis.yaml
│       ├── video_chunk_001__585.0sec.analysis.yaml
│       └── ...
└── video_eater_outputs/
    └── video_outputs/
        ├── full_video_analysis.yaml
        ├── video_youtube_description.md
        ├── video_video_analysis_report.md
        ├── video_video_analysis.json
        └── video_video_summary.txt
```
> Note - The video_eater_outputs are a level above the processed video/audio, to make it easy for you to nuke either when reprocessing. 

## Output file formats

### `full_video_analysis.yaml`

Combined analysis of all chunks. Contains the `FullVideoAnalysis` data model.

### `*_youtube_description.md`

Formatted for pasting into YouTube video descriptions. Includes chapters with timestamps, pull quotes, themes, topics, and takeaways.

### `*_video_analysis_report.md`

Detailed markdown report with full summaries and outlines.

### `*_video_analysis.json`

Machine-readable JSON export of the analysis.

### `*_video_summary.txt`

Plain text summary.

## Data models

### FullVideoAnalysis

The primary output model. Structure:

```python
class FullVideoAnalysis:
    summary: TranscriptSummaryPromptModel  # Overall summary
    chunk_analyses: list[ChunkAnalysisWithTimestamp]  # Per-chunk analysis
    themes: list[str]  # Main themes
    topics: list[TopicAreaPromptModel]  # Topic taxonomy
    takeaways: list[str]  # Key takeaways
    pull_quotes: list[PullQuoteWithTimestamp]  # Notable quotes
```

### TranscriptSummaryPromptModel

```python
class TranscriptSummaryPromptModel:
    transcript_title: str
    transcript_title_slug: str
    one_sentence_summary: str
    executive_summary: str  # 3-5 sentences
    topics_detailed_summary: str
    covered_topics_outline: list[TopicOutlineItem]
```

### ChunkAnalysisWithTimestamp

```python
class ChunkAnalysisWithTimestamp:
    starting_timestamp_string: str  # e.g. "123.45"
    summary: TranscriptSummaryPromptModel
    main_themes: list[str]
    key_takeaways: list[str]
    topic_areas: list[TopicAreaPromptModel]
    pull_quotes: list[PullQuoteWithTimestamp]
```

### PullQuoteWithTimestamp

```python
class PullQuoteWithTimestamp:
    quality: int  # 1-1000
    text_content: str
    reason_for_selection: str
    context_around_quote: str
    timestamp_seconds: float
```

### TopicAreaPromptModel

```python
class TopicAreaPromptModel:
    name: str
    category: str
    subject: str
    topic: str
    subtopic: str
    niche: str
    description: str
```

### VideoTranscript (intermediate format)

```python
class VideoTranscript:
    transcript_segments: list[TranscriptSegment]
    full_transcript_raw: str
    full_transcript_timestamps_srt: str
```

### TranscriptSegment

```python
class TranscriptSegment:
    text: str
    start: float  # seconds
    dur: float  # seconds (optional)
    end: float  # seconds (optional)
```

## Caching behavior

The pipeline caches intermediate outputs:

- Audio chunks: Skipped if `chunks/audio_chunks/` contains files matching the expected pattern
- Transcripts: Skipped if corresponding `.transcript.json` files exist
- Chunk analyses: Skipped if corresponding `.analysis.yaml` files exist
- Combined analysis: Skipped if `full_video_analysis.yaml` exists

Use the `force_*` config flags to override caching.

## Known issues and limitations

- YouTube downloads require browser cookies for authentication (the tool attempts to extract them automatically from installed browsers)
- Local Whisper transcription requires a GPU with sufficient VRAM for larger models
- The LLM analysis quality depends heavily on the model used and prompt details - update prompts in `video_eater/core/ai_processors/ai_prompt_models.py` and reprocess with different instructions
- Pull quote timestamp matching uses fuzzy matching and may be inaccurate
- Error handling is inconsistent across the codebase
- No automated tests

gl;hf! 