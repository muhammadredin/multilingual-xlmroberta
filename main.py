import logging
import logging.handlers
import queue
import threading
import time
import urllib.request
import os
import torch
import pyaudio
import re
import sys

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

TEXT_BUCKET = []

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

language_code = "id-ID"  # BCP-47 language tag for Indonesian
client = speech.SpeechClient.from_service_account_file('../key.json')
config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=RATE,
    language_code=language_code,
)
streaming_config = speech.StreamingRecognitionConfig(
    config=config, interim_results=True
)
button_thread = None
stream = None

@st.cache_resource
def get_models():
    # it may be necessary for other frameworks to cache the model
    # seems pytorch keeps an internal state of the conversation
    model_dir = "../my_model/xlm-roberta-model"
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

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self: object, rate: int, chunk: int) -> None:
        """The audio -- and generator -- is guaranteed to be on the main thread."""
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self: object) -> object:
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False
        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> None:
        """Closes the stream, regardless of whether the connection was lost or not."""
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self: object,
        in_data: object,
        frame_count: int,
        time_info: object,
        status_flags: object,
    ) -> object:
        """Continuously collect data from the audio stream, into the buffer.

        Args:
            in_data: The audio data as a bytes object
            frame_count: The number of frames captured
            time_info: The time information
            status_flags: The status flags

        Returns:
            The audio data as a bytes object
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
        """Generates audio chunks from the stream of audio data in chunks.

        Args:
            self: The MicrophoneStream object

        Returns:
            A generator that outputs audio chunks.
        """
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)
            
def listen_print_loop(responses: object, status, text) -> None:
    global TEXT_BUCKET
    
    listener_activated = False
    num_chars_printed = 0

    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue
        transcript = result.alternatives[0].transcript
        overwrite_chars = " " * (num_chars_printed - len(transcript))
        
        if not listener_activated:
            status.write("Listener Activated")
            listener_activated = True

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()
            num_chars_printed = len(transcript)
        else:
            text.markdown(transcript + overwrite_chars)
            TEXT_BUCKET.append(transcript)
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                status.write("Exiting..")
                break
            num_chars_printed = 0

def record_thread(stream: MicrophoneStream, streaming_config: speech.StreamingRecognitionConfig, status, text) -> None:
    with stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )
        responses = client.streaming_recognize(streaming_config, requests)
        listen_print_loop(responses, status, text)

def start_recording(status, text):
    global stream
    global button_thread
    stream = MicrophoneStream(RATE, CHUNK)
    status.write("Recording started.")
    button_thread = threading.Thread(target=record_thread, args=(stream, streaming_config, status, text))
    button_thread.start()

def stop_recording(status):
    global stream
    global button_thread
    status.write("Stopping Record...")
    if stream is not None:
        stream.__exit__(None, None, None)
    if button_thread is not None:
        button_thread.join()
    time.sleep(1)
    status.write("")

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
    app_mode = st.selectbox("Choose the app mode", [sound_classifier, text_classifier])

    if app_mode == sound_classifier:
        app_sst()
    elif app_mode == text_classifier:
        app_sst_with_video()

def app_sst():
    global stream
    global button_thread
    global config
    
    webrtc_ctx = webrtc_streamer(
        key="key",
        # mode=WebRtcMode.SENDONLY,
        # audio_receiver_size=1024,
        # rtc_configuration={
        #     "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        # },
        # media_stream_constraints={"video": False, "audio": True},
    )
    
    status_indicator = st.empty()
    text_output = st.empty()

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    status_indicator.write(webrtc_ctx.state.playing)
    if not webrtc_ctx.state.playing:
        return
    
    status_indicator.write("Recording started.")
    while True:
        if webrtc_ctx.audio_receiver:
            if stream is None:
                stream = MicrophoneStream(RATE, CHUNK)
                status_indicator.write("Recording started.")
                button_thread = threading.Thread(target=record_thread, args=(stream, streaming_config, status_indicator, text_output))
                button_thread.start()

            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Running. Say something!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                    streaming_config.config.sample_rate_hertz
                )
                buffer = np.array(sound_chunk.get_array_of_samples())
                request = speech.StreamingRecognizeRequest(audio_content=buffer.tobytes())
                stream.write(request)

                responses = list(stream)
                for response in responses:
                    for result in response.results:
                        if result.alternatives:
                            text = result.alternatives[0].transcript
                            text_output.markdown(f"**Text:** {text}")
        else:
            status_indicator.write("AudioReceiver is not set. Abort.")
            break

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