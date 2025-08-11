import asyncio
import json
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from openai.types.audio import  TranscriptionVerbose

from helpers.ai_processors.base_processor import BaseProcessor
from helpers.transcript_models import ProcessedTranscript
from helpers.yt_prompts import CLEANUP_TRANSCRIPT_SYSTEM_PROMPT
logger = logging.getLogger(__name__)

class OpenaiTranscriptCleaner(BaseProcessor):
    """Responsible for cleaning raw transcripts."""


    async def process_transcript(self,
                           transcript: TranscriptionVerbose
                           ) -> ProcessedTranscript:
        """Clean an individual transcript chunk."""

        logger.info(f"Sending transcript request to OpenAI ...")
        processed_transcript = await self.make_openai_json_mode_ai_request(
            system_prompt=CLEANUP_TRANSCRIPT_SYSTEM_PROMPT,  # noqa: F821
            input_data=transcript.model_dump(exclude="duration, language, words"),
            output_model=ProcessedTranscript
        )

        return processed_transcript

if __name__ == "__main__":
    _transcript_path = r"C:\Users\jonma\Sync\2025-08-07-JSM-Livestream-RAW_transcription.json"
    from pathlib import Path
    if not Path(_transcript_path).exists():
        raise FileNotFoundError(f"Transcript file not found: {_transcript_path}")
    _cleaner = OpenaiTranscriptCleaner()

    _og_transcript = TranscriptionVerbose(**json.loads(Path(_transcript_path).read_text()))
    print(f"loaded transcript with {len(_og_transcript.segments)} segments and {len(_og_transcript.words)} words, Transcript duration: {round(_og_transcript.duration, 2)} seconds")
    _processed = asyncio.run(_cleaner.process_transcript(_og_transcript))
    print(_processed)
    Path(str(_transcript_path).replace(".json",".cleaned.json")).write_text(_processed.model_dump_json(indent=2))
    print('done')



"Okay, that looks better. I think I was recording or streaming at 4K, which is too much. Do I have sound? Probably. Hello, hello. Hello, hello. Hello, hello, hello, hello, hello, hello. Looks like I have sound. That seems good. I think we're good. I think we're going. Oh man, it's been a second since I've done this. It's like a whole headspace that I need to get back into. I want to be looking at OBS. I think I don't need it. I want to look at the stats for OBS, which look like this and are doing fine right now. I will put those up there. I guess I can put this on top of that because I need to see chat, right? Let's see, what am I doing? Okay, I can't see chat through that. I think I'm here. Do I get to see who? Oh, okay, I think I am okay. So, you're probably wondering why I called you all here today. Yeah, I think we're doing fine. Is there anything else weird that happens when I do that? It's been so long since I've clicked the stream button that I just kind of don't remember what the process is. I'm also afraid of any automated things I may have set up years ago and then forgotten about. I don't think any of that's triggering. Okay, let's go ahead and get started. What are we doing? I'll start drawing on this. Okay, so let me reset the Wacom mapping to that one. Now I can draw like that. Let's see. So, one. One thousand, two. One thousand, three. One thousand. Okay, so like four, one, four one thousand. Four seconds of lag roughly between when I do something in real life and when it shows up on the screen, which is about what I have come to expect. Such a strange little headspace. You're just like sitting alone in a room with a couple of lights blinking and you're broadcasting your life out into the world. So, today we're going to talk about a variety of things. I'm going to give an update on FreeMoCap as a project, just kind of like talk about where we're at with it. I'm going to show off some data that we got recently that represents a high watermark of what we can do with the current status of FreeMoCap. I guess it's not technically the current status because it's using new stuff, but it's like, hey, here's the data we can produce. I'll talk a little bit about the roadmap to 2.0, which is very exciting because FreeMoCap has been chugging along and is in V1 status. It's doing a good job, but we're better now. We know more, we can do more, and I am very excited to take the whole current code base and throw it into the garbage and replace it with a complete new from scratch refactor, which is ongoing. Most importantly to me, at an emotional and spiritual level, is real-time processing. After that, I don't want to spend too much time on this, and then I want to spend most of the time here talking about the current state of SkellyCam 2.0, update demo, and et cetera. Because basically SkellyCam, which is the camera part of FreeMoCap, is more or less working. Obviously, there's always more that can be done. It works, and it is the basis upon which the 2.0 software will be built. So, the fact that SkellyCam is working means that the rest of the situation should hopefully resolve itself without too much difficulty. Let's go ahead and get started. The whole FreeMoCap project, I don't know how long it's been since I've done one of these updates around probably years. And so I don't remember where to start. The situation as it stands right now is that FreeMoCap is chugging along. We have an existing version of the software. We've got something like three thousand stars on GitHub. We've got a Discord server with about 2000 people in it. It's active now. People are in there doing stuff, talking about the work they're doing. There are people asking questions and others helping them out. It's really amazing. We have received, from the little anonymous data send feature for all the people who have not unchecked that box, I think last I checked, close to 8500 unique IP addresses, which are sort of proxied to hypothetical unique users and from 120 or 130 countries. That's the very exciting part for me. So, we have a lot of stuff going on. Yeah, there are two versions of the software right now, or I guess the two main thrusts of the project. There's the existing 1.0, which has been in use if you do pip install for FreeMoCap, that's the one you get. It's probably 1.6 something at this point. A lot of the work we've been doing in there is a combination of maintenance work. We're not really adding a lot of new functionality to it. We're mostly fixing bugs and making sure that people can use it in the main user base. There are a lot of little parts I can talk about later. But mostly it's been, you know, it's gone through a lot of changes, but it's roughly in the same architecture that I was using when I first wrote this code years ago, which is a bunch of PyQt, Python Qt GUI. It works, but it's clunky. It's kind of like the first one of these apps that I've really made, so mistakes were made that limit some of the development. But it's plateaued now, and there are some things that should be easy fixes, but they're not, just because of the structure of the code. Additionally, there is the not yet fully extant version. Sorry, one second. Yeah, I don't know what's going on there. Somebody's having a, what's going on? Sorry, one second. I lost my momentum. I didn't really have any. Okay. Hello. Welcome. Welcome NeonXDef. I don't know how to look at this. I wonder what just happened. I just saw the viewer count pop up to five, which is plenty. Hello. I don't know if there's a way to see a list of people in there. I'm just curious if those five people are actually people or if they are weird bots. So, this is what we're going to be talking about today. I guess I'm kind of lost. Okay, let's move on. Yes, so FreeMoCap 1.0 is extant and exists and works okay. It does its job decently, but it's plateaued in its development."

