"""
main.py
"""

# Standard library imports
import glob
import os
import time
import html
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Tuple, Optional

# Third-party imports
import gradio as gr
import random
from loguru import logger
from pypdf import PdfReader
from pydub import AudioSegment

# Local imports
from constants import (
    APP_TITLE,
    AUDIO_FILE_EXTENSION,
    CHARACTER_LIMIT,
    ERROR_MESSAGE_NOT_PDF,
    ERROR_MESSAGE_NO_INPUT,
    ERROR_MESSAGE_READING_PDF,
    ERROR_MESSAGE_TOO_LONG,
    GRADIO_CACHE_DIR,
    GRADIO_CLEAR_CACHE_OLDER_THAN,
    SPEECH_FORMAT,
    SPEECH_RATE,
    UI_ALLOW_FLAGGING,
    UI_API_NAME,
    UI_CACHE_EXAMPLES,
    UI_CONCURRENCY_LIMIT,
    UI_DESCRIPTION,
    UI_EXAMPLES,
    UI_INPUTS,
    UI_OUTPUTS,
    UI_SHOW_API,
)
from prompts import (
    LANGUAGE_MODIFIER,
    LENGTH_MODIFIERS,
    QUESTION_MODIFIER,
    SYSTEM_PROMPT,
    TONE_MODIFIER,
)
from schema import ShortDialogue, MediumDialogue
from utils import generate_script, parse_url

from dotenv import load_dotenv
load_dotenv()

import azure.cognitiveservices.speech as speechsdk
import os
speech_key = os.getenv('SPEECH_KEY')
service_region = os.getenv('SPEECH_REGION')

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
# Set audio output format to highest quality uncompressed 48kHz 16-bit WAV
speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff48Khz16BitMonoPcm)

import random
import string

def generate_random_filename(length=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

def escape_ssml_text(text: str) -> str:
    """Escape special characters for SSML/XML content."""
    # Use html.escape to handle &, <, > and then manually handle quotes
    escaped = html.escape(text, quote=False)
    # Additional escaping for SSML if needed
    return escaped

def determine_break_strength(text: str, line_index: int, total_lines: int) -> str:
    """Determine SSML break strength based on content impact and context."""
    text_lower = text.lower().strip()
    
    # Strong breaks for high-impact content
    if any(keyword in text_lower for keyword in [
        'important', 'urgent', 'breaking', 'announcement', 'attention', 
        'warning', 'alert', 'critical', 'emergency', 'deadline'
    ]):
        return "strong"
    
    # Medium breaks for transitional phrases and conclusions
    if any(phrase in text_lower for phrase in [
        'in other news', 'moving on', 'next up', 'meanwhile', 'however',
        'in conclusion', 'to summarize', 'finally', 'lastly', 'that wraps up'
    ]):
        return "medium"
    
    # Medium break for questions to audience
    if text_lower.endswith('?') and any(word in text_lower for word in [
        'you', 'your', 'wondering', 'think', 'imagine'
    ]):
        return "medium"
    
    # Strong break at the end of intro (first few lines)
    if line_index < 3 and any(phrase in text_lower for phrase in [
        'welcome', "today's", 'this is', "i'm", 'hello'
    ]):
        return "strong"
    
    # Strong break for outro/conclusion (last few lines)
    if line_index >= total_lines - 3:
        return "strong"
    
    # Medium break for end of sentences with exclamation
    if text_lower.endswith('!'):
        return "medium"
    
    # Weak break for regular statements ending with period
    if text_lower.endswith('.'):
        return "weak"
    
    # Medium break for statements ending with comma (continuation)
    if text_lower.endswith(','):
        return "weak"
    
    # Default to weak break
    return "weak"

def generate_podcast(
    files: List[str],
    url: Optional[str],
    text_input: Optional[str],
    question: Optional[str],
    tone: Optional[str],
    length: Optional[str],
    language: str
) -> Tuple[str, str]:
    """Generate the audio and transcript from the PDFs, URL, and/or direct text input."""

    text = ""

    # Check if at least one input is provided
    if not files and not url and not text_input:
        raise gr.Error(ERROR_MESSAGE_NO_INPUT)

    # Process PDFs if any
    if files:
        for file in files:
            if not file.lower().endswith(".pdf"):
                raise gr.Error(ERROR_MESSAGE_NOT_PDF)

            try:
                with Path(file).open("rb") as f:
                    reader = PdfReader(f)
                    text += "\n\n".join([page.extract_text() for page in reader.pages])
            except Exception as e:
                raise gr.Error(f"{ERROR_MESSAGE_READING_PDF}: {str(e)}")

    # Process URL if provided
    if url:
        try:
            url_text = parse_url(url)
            text += "\n\n" + url_text
        except ValueError as e:
            raise gr.Error(str(e))

    # Process direct text input if provided
    if text_input:
        text += "\n\n" + text_input.strip()

    # Check total character count
    if len(text) > CHARACTER_LIMIT:
        raise gr.Error(ERROR_MESSAGE_TOO_LONG)

    # Modify the system prompt based on the user input
    modified_system_prompt = SYSTEM_PROMPT

    if question:
        modified_system_prompt += f"\n\n{QUESTION_MODIFIER} {question}"
    if tone:
        modified_system_prompt += f"\n\n{TONE_MODIFIER} {tone}."
    if length:
        modified_system_prompt += f"\n\n{LENGTH_MODIFIERS[length]}"
    if language:
        modified_system_prompt += f"\n\n{LANGUAGE_MODIFIER} {language}."

    # Call the LLM
    if length == "Short (1-2 min)":
        llm_output = generate_script(modified_system_prompt, text, ShortDialogue)
    else:
        llm_output = generate_script(modified_system_prompt, text, MediumDialogue)

    logger.info(f"Generated dialogue: {llm_output}")

    # Process the dialogue
    audio_segments = []
    transcript = ""
    total_characters = 0

    voice: dict[str, dict[str, str]] = {
        "English": {
            "host": "en-us-Ava:DragonHDLatestNeural"
        },
        "German": {
            "host": "de-DE-Seraphina:DragonHDLatestNeural"
        },
        "French": {
            "host": "fr-FR-Vivienne:DragonHDLatestNeural"
        }
    }

    ssml = "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='en-US'>"

    total_dialogue_lines = len(llm_output.dialogue)
    
    for line_index, line in enumerate(llm_output.dialogue):
        escaped_text = escape_ssml_text(line.text)
        speaker = f"**Host**: {line.text}"
        
        # Determine break strength based on content impact
        break_strength = determine_break_strength(line.text, line_index, total_dialogue_lines)
        
        # Enhanced SSML with additional prosody controls and dynamic breaks
        ssml += f"\n<voice name='{voice[language]['host']}'><prosody rate='{SPEECH_RATE}' pitch='+0%' volume='+0%'><mstts:express-as style='friendly' styledegree='1.0'>{escaped_text}</mstts:express-as></prosody><break strength='{break_strength}'/></voice>"
        transcript += speaker + "\n\n"

    ssml += "</speak>"

    # Export the combined audio to a temporary file
    temporary_directory = GRADIO_CACHE_DIR
    os.makedirs(temporary_directory, exist_ok=True)

    logger.info(f"Generating audio using Azure AI Speech Service.")
    random_filename = generate_random_filename() + AUDIO_FILE_EXTENSION
    temporary_file = f"{temporary_directory}{random_filename}"
    
    # Save SSML content to a text file with the same name
    ssml_filename = random_filename.replace(AUDIO_FILE_EXTENSION, ".ssml.txt")
    ssml_file_path = f"{temporary_directory}{ssml_filename}"
    
    try:
        with open(ssml_file_path, 'w', encoding='utf-8') as ssml_file:
            # Write metadata header
            ssml_file.write(f"<!-- SSML File for Audio: {random_filename} -->\n")
            ssml_file.write(f"<!-- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')} -->\n")
            ssml_file.write(f"<!-- Language: {language} -->\n")
            ssml_file.write(f"<!-- Voice: {voice[language]['host']} -->\n")
            ssml_file.write(f"<!-- Speech Rate: {SPEECH_RATE} -->\n")
            ssml_file.write(f"<!-- Content Length: {len(ssml)} characters -->\n")
            ssml_file.write(f"<!-- Dialogue Lines: {len(llm_output.dialogue)} -->\n")
            ssml_file.write(f"<!-- Features: Dynamic break strength analysis -->\n\n")
            
            # Write break strength analysis
            ssml_file.write("<!-- Break Strength Analysis: -->\n")
            for i, line in enumerate(llm_output.dialogue):
                break_str = determine_break_strength(line.text, i, len(llm_output.dialogue))
                ssml_file.write(f"<!-- Line {i+1}: {break_str} break - \"{line.text[:50]}{'...' if len(line.text) > 50 else ''}\" -->\n")
            ssml_file.write("\n")
            
            # Write the actual SSML content
            ssml_file.write(ssml)
            
        logger.info(f"SSML content saved to: {ssml_file_path}")
    except Exception as e:
        logger.warning(f"Failed to save SSML file: {e}")
        # Continue with audio generation even if SSML save fails

    audio_output = speechsdk.audio.AudioOutputConfig(filename=temporary_file)

    # Creates a speech synthesizer using the Azure Speech Service.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)

    # Log the SSML for debugging
    logger.info(f"SSML content length: {len(ssml)} characters")
    logger.debug(f"SSML content: {ssml[:500]}...")  # Log first 500 chars for debugging

    # Synthesizes the received text to speech.
    result = speech_synthesizer.speak_ssml_async(ssml).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesis was successful. Audio was written to '{}'".format(temporary_file))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
        print("Did you update the subscription info?")

    # Delete any files in the temp directory that end with .wav and are over a day old
    for file in glob.glob(f"{temporary_directory}*{AUDIO_FILE_EXTENSION}"):
        if (
            os.path.isfile(file)
            and time.time() - os.path.getmtime(file) > GRADIO_CLEAR_CACHE_OLDER_THAN
        ):
            os.remove(file)
    
    # Also clean up old SSML files
    for file in glob.glob(f"{temporary_directory}*.ssml.txt"):
        if (
            os.path.isfile(file)
            and time.time() - os.path.getmtime(file) > GRADIO_CLEAR_CACHE_OLDER_THAN
        ):
            os.remove(file)

    logger.info(f"Generated {temporary_file}")

    return temporary_file, transcript


demo = gr.Interface(
    title=APP_TITLE,
    description=UI_DESCRIPTION,
    fn=generate_podcast,
    inputs=[
        gr.File(
            label=UI_INPUTS["file_upload"]["label"],  # Step 1: File upload
            file_types=UI_INPUTS["file_upload"]["file_types"],
            file_count=UI_INPUTS["file_upload"]["file_count"],
        ),
        gr.Textbox(
            label=UI_INPUTS["url"]["label"],  # Step 2: URL
            placeholder=UI_INPUTS["url"]["placeholder"],
        ),
        gr.Textbox(
            label=UI_INPUTS["text_input"]["label"],  # Step 3: Direct text input
            placeholder=UI_INPUTS["text_input"]["placeholder"],
            lines=UI_INPUTS["text_input"]["lines"],
        ),
        gr.Textbox(label=UI_INPUTS["question"]["label"]),  # Step 4: Question
        gr.Dropdown(
            label=UI_INPUTS["tone"]["label"],  # Step 5: Tone
            choices=UI_INPUTS["tone"]["choices"],
            value=UI_INPUTS["tone"]["value"],
        ),
        gr.Dropdown(
            label=UI_INPUTS["length"]["label"],  # Step 6: Length
            choices=UI_INPUTS["length"]["choices"],
            value=UI_INPUTS["length"]["value"],
        ),
        gr.Dropdown(
            choices=UI_INPUTS["language"]["choices"],  # Step 7: Language
            value=UI_INPUTS["language"]["value"],
            label=UI_INPUTS["language"]["label"],
        ),
    ],
    outputs=[
        gr.Audio(
            label=UI_OUTPUTS["audio"]["label"], format=UI_OUTPUTS["audio"]["format"]
        ),
        gr.Markdown(label=UI_OUTPUTS["transcript"]["label"]),
    ],
    allow_flagging=UI_ALLOW_FLAGGING,
    api_name=UI_API_NAME,
    theme=gr.themes.Ocean(),
    concurrency_limit=UI_CONCURRENCY_LIMIT,
    examples=UI_EXAMPLES,
    cache_examples=UI_CACHE_EXAMPLES,
)

if __name__ == "__main__":
    demo.launch(show_api=UI_SHOW_API, favicon_path='favicon.png')
