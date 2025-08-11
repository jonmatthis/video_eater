import re
from pydantic import BaseModel, Field


class YoutubeVideoMetadata(BaseModel):
    title: str
    author: str
    view_count: str
    description: str
    publish_date: str
    channel_id: str
    duration: str
    like_count: str = None
    tags: str = None

    @property
    def clean_title(self) -> str:
        return re.sub(r'[^a-zA-Z0-9 ]', '', self.title).replace(' ', '_').lower()

