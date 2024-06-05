from datetime import datetime
import json
import os
import sys

import boto3
from dotenv import load_dotenv

from griptape.drivers import ElevenLabsTextToSpeechDriver
from griptape.engines import TextToSpeechEngine
from griptape.rules import Rule
from griptape.structures import Agent
from griptape.tools import WebScraper, AwsS3Client
from griptape.tools.text_to_speech_client.tool import TextToSpeechClient

load_dotenv()

# Example input:
# {
#     "text_url": "https://www.griptape.ai/blog/announcing-griptape",
#     "languages": [
#         "finnish",
#         "korean"
#     ],
#     "output_bucket": "griptape-andrew-test-bucket"
# }

input_obj = json.loads(sys.argv[1])
input_text_url = input_obj["text_url"]
langs = input_obj["languages"]
output_bucket = input_obj["output_bucket"]

session = boto3.Session(
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    aws_session_token=os.environ["AWS_SESSION_TOKEN"],
)

tts_engine = TextToSpeechEngine(
    text_to_speech_driver=ElevenLabsTextToSpeechDriver(
        api_key=os.environ["ELEVEN_LABS_API_KEY"],
        model="eleven_multilingual_v2",
        voice="Daniel",
    )
)

agent = Agent(
    rules=[
        Rule("Translate the text into the each language automatically without relying on external tools or services."),
        Rule("Provide already translated text as an input to each text-to-speech task."),
        Rule("Remove all Markdown formatting for headers, links, etc. from the translated text."),
        Rule("Preserve paragraph breaks to ensure that the speech output is paced naturally."),
        Rule("Upload the audio content using the upload_memory_artifacts_to_s3 activity."),
        Rule("Upload object keys with the following name format: <YY>_<MM>_<DD>_<language>.mp3."),
    ],
    tools=[
        WebScraper(),
        TextToSpeechClient(engine=tts_engine, off_prompt=True),
        AwsS3Client(session=session),
    ]
)

prompt = f"""
Translate the text content at the provided URL to the requested languages, then generate speech for each language 
before uploading each output audio artifact to the provided S3 bucket.

Input URL: {input_text_url}
Languages: {langs}
S3 Bucket: {output_bucket}
Today's Date: {datetime.now().strftime("%Y-%m-%d")}
"""

prompt += """
Respond with a JSON object indicating the success of the task and the S3 object key for each language, this format:
{
    "success": true,
    "outputs": [
        {
            "language": "<language>",
            "url": "<S3 object URL>"
        }
    ]
} 
"""

agent.run(prompt)
