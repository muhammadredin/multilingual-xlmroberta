import logging
import logging.handlers
import queue
import threading
import time
import urllib.request
import os
import torch
import re
import sys
import base64

from google.cloud import speech

from collections import deque
from pathlib import Path
from typing import List

import av
import numpy as np
import pydub
import streamlit as st
from twilio.rest import Client
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

from streamlit_webrtc import WebRtcMode, webrtc_streamer

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

# Get the directory where main.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to file.txt
file_path = os.path.join(current_dir, "key.json")

client = speech.SpeechClient.from_service_account_file(file_path)

# TEXT_BUCKET = []

# # Audio recording parameters
# RATE = 16000
# CHUNK = int(RATE / 10)  # 100ms

# language_code = "id-ID"  # BCP-47 language tag for Indonesian
# client = speech.SpeechClient.from_service_account_file('key.json')
# config = speech.RecognitionConfig(
#     encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#     sample_rate_hertz=RATE,
#     language_code=language_code,
# )
# streaming_config = speech.StreamingRecognitionConfig(
#     config=config, interim_results=True
# )
# button_thread = None
# stream = None

@st.cache_resource
def get_models():
    # it may be necessary for other frameworks to cache the model
    # seems pytorch keeps an internal state of the conversation
    app_directory = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(app_directory, "model")
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_dir)
    model = XLMRobertaForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

# This code is based on https://github.com/whitphx/streamlit-webrtc/blob/c1fe3c783c9e8042ce0c95d789e833233fd82e74/sample_utils/turn.py
@st.cache_data  # type: ignore
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """

    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers

def main():
    st.header("Multilingual Speech-Text-based Toxic Comment Classifier")
    st.markdown(
        """
Welcome To Our Multilingual Speech-text-based Toxic Comment Classifier

This Project is created based on the XLM-Roberta Cross-lingual Algorithm for The World Invention Competition and Exhibiton 2023
"""
    )

    sound_classifier = "Sound-based Classifier"
    text_classifier = "Text-based Classifier"
    app_mode = st.selectbox("Choose the app mode", [text_classifier, sound_classifier])

    if app_mode == sound_classifier:
        app_sst()
    elif app_mode == text_classifier:
        app_sst_with_video()

def app_sst():
#     global streamhttps://github.com/muhammadredin/multilingual-xlmroberta/blob/main/main.py
#     global button_thread
#     global config
    global client

    st.title("Sorry, this feature is still on maintenance")   

    audio = st.file_uploader("Choose an audio file")
    
    if audio is not None:
        # To read file as bytes:
        audio_path = "audio_file.mp3"

        audio_data = audio.read()
        
        with open(audio_path, "wb") as f:
            f.write(audio_data)

        encoded_audio_data = base64.b64encode(audio_data).decode()
        
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='en-US'
        )

        audio_file = speech.RecognitionAudio(content=encoded_audio_data)

        response = client.recognize(
            config=config,
            audio=audio_file
        )

        st.title(response.results.alternatives[0])
            
#     webrtc_ctx = webrtc_streamer(
#         key="key",
#         # mode=WebRtcMode.SENDONLY,
#         # audio_receiver_size=1024,
#         # rtc_configuration={
#         #     "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#         # },
#         # media_stream_constraints={"video": False, "audio": True},
#     )
    
#     status_indicator = st.empty()
#     text_output = st.empty()

#     streaming_config = speech.StreamingRecognitionConfig(
#         config=config, interim_results=True
#     )

#     status_indicator.write(webrtc_ctx.state.playing)
#     if not webrtc_ctx.state.playing:
#         return
    
#     status_indicator.write("Recording started.")
#     while True:
#         if webrtc_ctx.audio_receiver:
#             if stream is None:
#                 stream = MicrophoneStream(RATE, CHUNK)
#                 status_indicator.write("Recording started.")
#                 button_thread = threading.Thread(target=record_thread, args=(stream, streaming_config, status_indicator, text_output))
#                 button_thread.start()

#             sound_chunk = pydub.AudioSegment.empty()
#             try:
#                 audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
#             except queue.Empty:
#                 time.sleep(0.1)
#                 status_indicator.write("No frame arrived.")
#                 continue

#             status_indicator.write("Running. Say something!")

#             for audio_frame in audio_frames:
#                 sound = pydub.AudioSegment(
#                     data=audio_frame.to_ndarray().tobytes(),
#                     sample_width=audio_frame.format.bytes,
#                     frame_rate=audio_frame.sample_rate,
#                     channels=len(audio_frame.layout.channels),
#                 )
#                 sound_chunk += sound

#             if len(sound_chunk) > 0:
#                 sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
#                     streaming_config.config.sample_rate_hertz
#                 )
#                 buffer = np.array(sound_chunk.get_array_of_samples())
#                 request = speech.StreamingRecognizeRequest(audio_content=buffer.tobytes())
#                 stream.write(request)

#                 responses = list(stream)
#                 for response in responses:
#                     for result in response.results:
#                         if result.alternatives:
#                             text = result.alternatives[0].transcript
#                             text_output.markdown(f"**Text:** {text}")
#         else:
#             status_indicator.write("AudioReceiver is not set. Abort.")
#             break

    # col1, col2 = st.columns([0.5, 2])
    
    # with col1:
    #     if st.button('Start Recording'):
    #         if button_thread is None or not button_thread.is_alive():
    #             start_recording(status_indicator, text_output)
    # with col2:
    #     if st.button('Stop Recording'):
    #         stop_recording(status_indicator)

if "history" not in st.session_state:
    st.session_state.history = []

def generate_answer(text: str):
    tokenizer, model = get_models()
    user_message = text
    
    encoded_dict = tokenizer.encode_plus(
                        user_message,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_id = encoded_dict['input_ids']
        
    # And its attention mask (simply differentiates padding from non-padding).
    attention_mask = encoded_dict['attention_mask']
    input_id = torch.LongTensor(input_id)
    attention_mask = torch.LongTensor(attention_mask)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_id = input_id.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(input_id, token_type_ids=None, attention_mask=attention_mask)
    # print(outputs[0])
    logits = outputs[0]
    index = logits.argmax()
    
    return index

def app_sst_with_video():  
    st.title("Chat Room")

    user_message = st.chat_input("Chat something", key="input_text")
    st.session_state.history.append({"message": user_message, "response": "user"})

    if len(st.session_state.history) > 1:
        for i, chat in enumerate(st.session_state.history):
            if chat["message"] == None:
                continue
            if generate_answer(chat["message"]) == 1:
                message = st.chat_message("assistant")
                message.write("TOXIC COMMENT NOT ALLOWED")
                st.session_state.history.pop(i)
            else:
                message = st.chat_message(chat["response"])
                message.write(chat["message"])

if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
