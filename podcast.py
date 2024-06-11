from datetime import datetime
import json
import os
import sys

import boto3
from dotenv import load_dotenv

from griptape.drivers import OpenAiTextToSpeechDriver
from griptape.engines import TextToSpeechEngine
from griptape.rules import Rule
from griptape.structures import Agent
from griptape.tools import WebScraper, AwsS3Client, DateTime, TextToSpeechClient, FileManager

load_dotenv()

# Example input:
# {
#     "sources": [
#         "https://adn.com",
#         "https://nytimes.com",
#         "https://seattletimes.com",
#         "https://news.ycombinator.com"
#     ]
# }

# input_obj = json.loads(sys.argv[1])
input_obj = {
    "sources": [
        "https://en.wikipedia.org/wiki/Portal:Current_events",
        "https://old.reddit.com/r/news/top/",
        "https://www.reuters.com/",
        "https://text.npr.org/"
    ]
}

sources = input_obj["sources"]
output_bucket = os.environ["OUTPUT_AWS_S3_BUCKET_NAME"]

session = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    aws_session_token=os.environ["AWS_SESSION_TOKEN"],
)

# tts_engine = TextToSpeechEngine(
#     text_to_speech_driver=ElevenLabsTextToSpeechDriver(
#         api_key=os.environ["ELEVEN_LABS_API_KEY"],
#         model="eleven_multilingual_v2",
#         voice="Rachel",
#     )
# )

tts_engine = TextToSpeechEngine(
    text_to_speech_driver=OpenAiTextToSpeechDriver(voice="nova"),
)

agent = Agent(
    rules=[
        Rule("Generate text that flows well when read literally."),
        Rule("Spell complex or uncommon words or names phonetically to assist in reading aloud."),
        Rule("Always spell all numbers and acronyms to ensure they are read correctly."),
        Rule("Avoid formatting the transcript with section headers or lists."),
        Rule("Combine summaries into one transcript used to generate the episode audio."),
        Rule("Only use the SpeechToText client once to generate the complete episode audio."),
        Rule("Provide a brief introduction and conclusion to the episode."),
        Rule("Use the FileManager's save_memory_content_to_disk activity to save the audio artifact."),
    ],
    tools=[
        WebScraper(),
        DateTime(),
        TextToSpeechClient(engine=tts_engine, off_prompt=True),
        # AwsS3Client(session=session),
        FileManager(),
    ]
)

# prompt = f"""
# Generate an episode of the Seattle Daily Update Podcast by scraping text content from the provided
# current events URLs.
#
# First, identify a list of about ten of the most important news items to cover in this episode. The topics selected
# should be those most important to residents of the Seattle, Washington area.
#
# Use the WebScraper to make followup requests to gain context on potential topics. Generate an executive summary for
# each news item. The summary must contain enough information for a listener to have an understanding of the topic.
# Combine these summaries into one complete episode transcript.
#
# Finally, submit the entire transcript to the TextToSpeechClient to generate the episode audio. Save the audio artifact
# stored in memory to a file named with the following: <YY>_<MM>_<DD>_seattle-daily-news-podcast.mp3.
#
# Sources: {sources}
# """

prompt = f"""
Generate an episode of the Daily Tidbit Podcast by scraping text content from the provided current events pages.

First, identify a list of about ten of the most important news items to cover in this episode. The topics selected
should be those most relevant for the day's update, like recent breaking news.

Use the WebScraper to make followup requests to gain context on potential topics. Generate an executive summary for
each news item. The summary must contain enough information for a listener to have an understanding of the topic,
including relevant names, opinions, and locations. Combine these summaries into one complete episode transcript. Save
this transcript as a file named with the following format: <YY>-<MM>-<DD>_daily-tidbit-podcast.txt.

Finally, submit the entire transcript to the TextToSpeechClient to generate the episode audio. Save the audio artifact
stored in memory to a file named with the following format: <YY>-<MM>-<DD>_daily-tidbit-podcast.mp3.

Sources: {sources}
"""

agent.run(prompt)
