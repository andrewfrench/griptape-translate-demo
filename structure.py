import json
import os
import sys

from dotenv import load_dotenv

from griptape.artifacts import TextArtifact
from griptape.drivers import ElevenLabsTextToSpeechDriver
from griptape.engines import TextToSpeechEngine
from griptape.structures import Workflow
from griptape.tools import WebScraper
from griptape.tasks import PromptTask, CodeExecutionTask, TextToSpeechTask, BaseTask, ToolTask


load_dotenv()

input_obj = json.loads(sys.argv[1])
# input_audio_file_url = input_obj["audio_file_url"]
input_text_url = input_obj["text_url"]
# "https://www.griptape.ai/blog/announcing-griptape"
langs = input_obj["languages"]
# ["english", "german", "french", "spanish"]

webscraper_task = ToolTask(
    f"Return the content of this blog post: {input_text_url}",
    tool=WebScraper(off_prompt=False),
    id="webscraper_task",
)

# load_audio_task = CodeExecutionTask(
#     run_fn=lambda _: AudioLoader().load(requests.get(input_audio_file_url).content),
#     id="load_audio_task",
# )
#
# transcription_task = AudioTranscriptionTask(
#     input=lambda task: task.parents[0].output,
#     id="transcription_task",
# )


def make_translation_task(lang: str) -> list[BaseTask]:
    lang_prompt = (f"Prepare the following text for {lang} speech synthesis. Translate the following text "
                   f"into {lang} and remove all Markdown formatting characters defining links, headings, "
                   f"images, and the like.")
    common_prompt = """

        {{ parent_outputs['webscraper_task'] }}
    """

    translate_task = PromptTask(
        f"""
        {lang_prompt}
        {common_prompt}
        """,
        id=f"translation_task_{lang}",
    )

    tts_task = TextToSpeechTask(
        lambda task: task.parents[0].output,
        output_file=f"/Users/andrew/Downloads/demolangs/blog_post_{lang}.mp3",
        text_to_speech_engine=TextToSpeechEngine(
            text_to_speech_driver=ElevenLabsTextToSpeechDriver(
                api_key=os.environ["ELEVEN_LABS_API_KEY"],
                model="eleven_multilingual_v2",
                voice="Rachel",
            )
        ),
        id=f"tts_task_{lang}",
    )

    return [translate_task, tts_task]


end_task = CodeExecutionTask(
    run_fn=lambda _: TextArtifact("Done."),
    id="end_task",
)

translation_tasks = [make_translation_task(lang) for lang in langs]

workflow = Workflow()
workflow.add_task(webscraper_task)
workflow.add_task(end_task)
for translation_task in translation_tasks:
    workflow.insert_tasks(webscraper_task, [translation_task[0]], end_task)
    workflow.insert_tasks(translation_task[0], [translation_task[1]], end_task)

workflow.run()
