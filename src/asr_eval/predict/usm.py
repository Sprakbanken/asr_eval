# # GCP Project to use when calling the Cloud Speech-to-Text API.
# GCP_PROJECT_ID='norwegian-language-bank' # @param

# # Auth commands
# # Set GCP project to use.
# !gcloud config set project "$GCP_PROJECT_ID"
# !gcloud auth login

# # Activate the Cloud Speech-to-Text v2 API
# !gcloud services enable speech.googleapis.com --project="$GCP_PROJECT_ID"

import io
import os

from google.api_core import exceptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

from dotenv import load_dotenv

load_dotenv()


class SpeechParameters(object):
    """Holds required parameters to create Speech-to-Text Recognizer.
    A Recognizer is a GCP resource that contains:
      * An identificator, provided by you.
      * A model to use in recognition requests.
      * A language code or locale.
    You can learn more about Recognizers here:
    https://cloud.google.com/speech-to-text/v2/docs/basics#recognizers
    """

    def __init__(self, model: str):
        # Recognizer Id. This allowes you to name the Recognizer.
        # Must be unique by GCP project/location.
        self.recognizer_id = "usm-sprakbanken-no"  # @param
        # Language code to use with this recognizer.
        self.locale = "no-NO"  # @param
        # Use the USM model. Don't change if you want to actually use the USM model.
        self.model = model
        # GCP project to interact with Cloud Speech-to-Text API.
        self.gcp_project = os.environ.get("GCP_PROJECT_ID")  # @param

    def base_recognizer_path(self):
        return f"projects/{self.gcp_project}/locations/us-central1"

    def full_recognizer_path(self):
        return f"{self.base_recognizer_path()}/recognizers/{self.recognizer_id}"


# @title Cloud Speech-to-Text Implementation.
class SpeechInterface(object):
    """Implementation of the Cloud Speech-to-Text API.

    Exposes CreateRecognizer and Recognize calls.
    """

    def __init__(self, speech_params: SpeechParameters):
        self.speech_params_ = speech_params
        self.speech_client_ = SpeechClient(
            client_options={
                "api_endpoint": "us-central1-speech.googleapis.com",
            }
        )
        self.recognizer_ = None

    def CreateRecognizer(self):
        """Creates a Recognizer if it doesn't exist.

        Args: None
        Returns: None
        """
        need_to_create_recognizer = False
        # Fetch recognizer, or create it if it doesn't exist.
        try:
            self.recognizer_ = self.speech_client_.get_recognizer(
                name=self.speech_params_.full_recognizer_path()
            )
        except exceptions.NotFound:
            need_to_create_recognizer = True
        except Exception as generic_ex:
            raise generic_ex

        # Create a Recognizer if it doesn't exist.
        if need_to_create_recognizer:
            print(f"Creating Recognizer ({self.speech_params_.full_recognizer_path()})")
            request = cloud_speech.CreateRecognizerRequest(
                parent=self.speech_params_.base_recognizer_path(),
                recognizer_id=self.speech_params_.recognizer_id,
                recognizer=cloud_speech.Recognizer(
                    language_codes=[self.speech_params_.locale],
                    model=self.speech_params_.model,
                ),
            )
            operation = self.speech_client_.create_recognizer(request=request)
            self.recognizer_ = operation.result()
            print(f"Recognizer {self.speech_params_.full_recognizer_path()} created.")
            return
        print(
            "No need to create Recognizer "
            f"({self.speech_params_.full_recognizer_path()}). It already exists: "
        )

    def Recognize(self, audio_file: str) -> cloud_speech.RecognizeResponse:
        """Calls Speech-to-Text Recognize with audio provided.

        Args: (string) audio_file: Audio file local path, or GCS URI to transcribe.
        Returns: cloud_speech.RecognizeResponse
        """
        recognition_config = cloud_speech.RecognitionConfig(auto_decoding_config={})
        recognition_request = cloud_speech.RecognizeRequest(
            recognizer=self.speech_params_.full_recognizer_path(),
            config=recognition_config,
        )
        if audio_file.startswith("gs://"):
            recognition_request.uri = audio_file
        else:
            with io.open(audio_file, "rb") as f:
                recognition_request.content = f.read()

        # Transcribes the audio into text
        response = self.speech_client_.recognize(request=recognition_request)
        return response
