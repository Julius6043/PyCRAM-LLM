from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import base64
import requests
import os

load_dotenv()

client = OpenAI()


def speech_to_text(speech_file_path):
    audio_file = open(speech_file_path, "rb")
    response = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    return response.text
