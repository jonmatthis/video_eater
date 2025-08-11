import asyncio
import logging

# from helpers.ai_yt_transcript_processor import AITranscriptProcessor
from youtube_transcript_api import YouTubeTranscriptApi

from helpers.ai_yt_transcript_processor import AITranscriptProcessor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(name)s | %(lineno)d | %(message)s',)
logger = logging.getLogger(__name__)

# Suppress some external loggers
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

def get_video_transcript(video_id: str) -> dict:
    youtube_transcript_api = YouTubeTranscriptApi()
    transcript = youtube_transcript_api.fetch(video_id=video_id)
    print(transcript)


async def run_video_eater_main(video_id:str, force_refresh: bool = False):
    try:
        logger.info("Starting YouTube lecture processing pipeline")

        # Extract transcripts from playlist
        video_transcript = get_video_transcript(video_id=video_id)
        logger.info(f"Processed video transcript for video ID: {video_id}, transcript length: {len(video_transcript.get('transcript', []))} entries")

        # AI processing (if you're keeping this)
        ai_processor = AITranscriptProcessor(force_refresh=force_refresh)
        await ai_processor.process_transcripts()

        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    YOUTUBE_ID = "GZl0apGwnxY" # [2025-08-07 - Livestream] Project check-in, Treadmill data exploration, and SkellyCam 2.0 updates - https://www.youtube.com/watch?v=GZl0apGwnxY
    logger.info(f"Using video ID: {YOUTUBE_ID}")
    asyncio.run(run_video_eater_main(video_id=YOUTUBE_ID, force_refresh=True))