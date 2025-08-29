# __main__.py (refactored)
import asyncio
from pathlib import Path
import click
import yaml

from video_eater.core.config_models import VideoProject, ProcessingConfig
from video_eater.core.pipeline import VideoProcessingPipeline
from video_eater.logging_config import PipelineLogger


@click.command()
@click.argument('video_paths', nargs=-1, type=click.Path(exists=True))
@click.option('--config', type=click.Path(), help='Config file path')
@click.option('--force-all', is_flag=True, help='Force reprocess everything')
@click.option('--log-level', default='INFO',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']))
def main(video_paths, config, force_all, log_level):
    """Process videos through the transcription and analysis pipeline."""

    # Load configuration
    if config:
        config_data = yaml.safe_load(Path(config).read_text())
        processing_config = ProcessingConfig(**config_data)
    else:
        processing_config = ProcessingConfig()

    if force_all:
        processing_config.force_chunk_audio = True
        processing_config.force_transcribe = True
        processing_config.force_analyze = True

    # Default videos if none provided
    if not video_paths:
        video_paths = [
            # r"C:\Users\jonma\Sync\videos\social-media-posts\2025-07-04-TikTok-GeneralStrike2028\testtest\2025-07-04-TikTok-GeneralStrike2028.mp4"
            r"\\jon-nas\jon-nas\videos\livestream_videos\2025-08-14-JSM-Livestream-Skellycam\2025-08-14-JSM-Livestream-Skellycam.mp4",
            # ... other default paths
        ]

    # Setup logging
    logger = PipelineLogger(log_level=log_level)

    # Process videos
    pipeline = VideoProcessingPipeline(config=processing_config, logger=logger)

    for video_path in video_paths:
        project = VideoProject(video_path=Path(video_path))
        result = asyncio.run(pipeline.process_video(project))

        print(result.summary_report())


if __name__ == "__main__":
    main()