import asyncio
import json
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from video_eater.core.ai_processors.base_processor import BaseProcessor



logger = logging.getLogger(__name__)

class OpenaiProcessor(BaseProcessor):
    pass 


