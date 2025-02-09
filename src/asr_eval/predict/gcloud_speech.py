from google.cloud import speech
import io
from dotenv import load_dotenv
import wave

load_dotenv()  # take environment variables


def google_cloud_transcribe(filepath):
    """Transcribe a file using Google Cloud STT.
    If no transcription is produced, "_" is returned"""

    with wave.open(filepath, "rb") as wave_file:
        sample_rate = wave_file.getframerate()
        num_channels = wave_file.getnchannels()

    gcp_speech_config = speech.RecognitionConfig(
        sample_rate_hertz=sample_rate,
        language_code="no-NO",
        model="latest_long",
        audio_channel_count=num_channels,
    )

    client = speech.SpeechClient()
    with io.open(filepath, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    try:
        response = client.recognize(config=gcp_speech_config, audio=audio)
        return " ".join(
            [result.alternatives[0].transcript for result in response.results]
        )
    except Exception as e:
        print(e)
        return "_"
