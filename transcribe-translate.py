import os

from griptape.artifacts import BaseArtifact, TextArtifact
from griptape.drivers import OpenAiAudioTranscriptionDriver
from griptape.drivers import ElevenLabsTextToSpeechDriver
from griptape.engines import TextToSpeechEngine
from griptape.structures import Workflow
from griptape.tasks import PromptTask, ToolTask, CodeExecutionTask, TextToSpeechTask, BaseTask
from griptape.tools.transcription_client.tool import TranscriptionClient


def print_return(artifact: BaseArtifact):
    print(artifact.value)
    return artifact


transcription_task = ToolTask(
    "Transcribe the audio file {{ args[0] }}.",
    tool=TranscriptionClient(
        off_prompt=False,
        driver=OpenAiAudioTranscriptionDriver(model="whisper-1")
    ),
    id="transcription_task",
)


def make_translation_task(lang: str) -> list[BaseTask]:
    lang_prompt = f"Translate the following text into {lang}."
    common_prompt = """

        {{ parent_outputs['transcription_task'] }}
    """

    revision_task = PromptTask(
        f"""
        {lang_prompt}
        {common_prompt}
        """,
        id=f"translation_task_{lang}",
    )

    tts_task = TextToSpeechTask(
        lambda task: task.parents[0].output,
        output_dir="demolangs/",
        text_to_speech_engine=TextToSpeechEngine(
            text_to_speech_driver=ElevenLabsTextToSpeechDriver(
                api_key=os.environ["ELEVEN_LABS_API_KEY"],
                model="eleven_multilingual_v2",
                voice="Matilda",
            )
        ),
        id=f"tts_task_{lang}",
    )

    return [revision_task, tts_task]


end_task = CodeExecutionTask(
    run_fn=lambda _: TextArtifact("Done."),
    id="end_task",
)

langs = ["german"]
translation_tasks = [make_translation_task(lang) for lang in langs]

workflow = Workflow()
workflow.add_task(transcription_task)
workflow.add_task(end_task)
for translation_task in translation_tasks:
    workflow.insert_tasks(transcription_task, [translation_task[0]], end_task)
    workflow.insert_tasks(translation_task[0], [translation_task[1]], end_task)

# print(workflow.to_graph())
workflow.run("demo_prompt_2.m4a")

