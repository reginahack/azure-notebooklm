"""
utils.py

Functions:
- generate_script: Get the dialogue from the LLM.
- call_llm: Call the LLM with the given prompt and dialogue format.
- parse_url: Parse the given URL and return the text content.
- generate_podcast_audio: Generate audio for podcast using TTS or advanced audio models.
"""

# Standard library imports
import time
from typing import Any, Union

# Third-party imports
import requests
from gradio_client import Client
from openai import OpenAI, AzureOpenAI
from pydantic import ValidationError
from scipy.io.wavfile import write as write_wav

import os
from dotenv import load_dotenv
load_dotenv()

from typing import Literal, List
from pydantic import BaseModel, Field


class DialogueItem(BaseModel):
    """A single dialogue item."""

    speaker: Literal["Host (Alice)", "Guest (Franck)"] = Field(
        ..., description="The speaker of the dialogue item, either 'Host (Alice)' or 'Guest (Franck)'"
    )
    text: str


class ShortDialogue(BaseModel):
    """The dialogue between the host and guest."""

    scratchpad: str
    name_of_guest: str
    dialogue: List[DialogueItem] = Field(
        ..., description="A list of dialogue items, typically between 11 to 17 items"
    )


class MediumDialogue(BaseModel):
    """The dialogue between the host and guest."""

    scratchpad: str
    name_of_guest: str
    dialogue: List[DialogueItem] = Field(
        ..., description="A list of dialogue items, typically between 29 to 39 items"
    )

JSON_RETRY_ATTEMPTS = 3

# Initialize clients
client = AzureOpenAI(
            api_key=os.environ['AZURE_OPENAI_KEY'],
            api_version=os.environ['AZURE_OPENAI_VERSION'],
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT']
        )

def generate_script(
    system_prompt: str,
    input_text: str,
    output_model: Union[ShortDialogue, MediumDialogue],
) -> Union[ShortDialogue, MediumDialogue]:
    """Get the dialogue from the LLM."""

    # Call the LLM
    response = call_llm(system_prompt, input_text, output_model)
    response_json = response.choices[0].message.content

    first_draft_dialogue = output_model.model_validate_json(response_json)

    # Validate the response
    for attempt in range(JSON_RETRY_ATTEMPTS):
        try:
            first_draft_dialogue = output_model.model_validate_json(response_json)
            break
        except ValidationError as e:
            if attempt == JSON_RETRY_ATTEMPTS - 1:  # Last attempt
                raise ValueError(
                    f"Failed to parse dialogue JSON after {JSON_RETRY_ATTEMPTS} attempts: {e}"
                ) from e
            error_message = (
                f"Failed to parse dialogue JSON (attempt {attempt + 1}): {e}"
            )
            # Re-call the LLM with the error message
            system_prompt_with_error = f"{system_prompt}\n\nPlease return a VALID JSON object. This was the earlier error: {error_message}"
            response = call_llm(system_prompt_with_error, input_text, output_model)
            response_json = response.choices[0].message.content
            first_draft_dialogue = output_model.model_validate_json(response_json)

    # Call the LLM a second time to improve the dialogue
    system_prompt_with_dialogue = f"{system_prompt}\n\nHere is the first draft of the dialogue you provided:\n\n{first_draft_dialogue}."

    # Validate the response
    for attempt in range(JSON_RETRY_ATTEMPTS):
        try:
            response = call_llm(
                system_prompt_with_dialogue,
                "Please improve the dialogue. Make it more natural and engaging.",
                output_model,
            )
            final_dialogue = output_model.model_validate_json(
                response.choices[0].message.content
            )
            break
        except ValidationError as e:
            if attempt == JSON_RETRY_ATTEMPTS - 1:  # Last attempt
                raise ValueError(
                    f"Failed to improve dialogue after {JSON_RETRY_ATTEMPTS} attempts: {e}"
                ) from e
            error_message = f"Failed to improve dialogue (attempt {attempt + 1}): {e}"
            system_prompt_with_dialogue += f"\n\nPlease return a VALID JSON object. This was the earlier error: {error_message}"
    return final_dialogue


def call_llm(system_prompt: str, text: str, dialogue_format: Any) -> Any:
    """Call the LLM with the given prompt and dialogue format."""
    response = client.beta.chat.completions.parse(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        model=os.environ["AZURE_OPENAI_MODEL"],
        max_tokens=16384,
        temperature= 0.1,
        response_format=dialogue_format,
    )
    return response
