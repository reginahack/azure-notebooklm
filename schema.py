"""
schema.py
"""

from typing import Literal, List

from pydantic import BaseModel, Field


class DialogueItem(BaseModel):
    """A single monologue item."""

    speaker: Literal["Host (Alice)"] = Field(
        ..., description="The speaker of the monologue item, always 'Host (Alice)'"
    )
    text: str

class ShortDialogue(BaseModel):
    """The monologue by the host."""

    scratchpad: str
    dialogue: List[DialogueItem] = Field(
        ..., description="A list of monologue items, typically between 11 to 17 items"
    )


class MediumDialogue(BaseModel):
    """The monologue by the host."""

    scratchpad: str
    dialogue: List[DialogueItem] = Field(
        ..., description="A list of monologue items, typically between 29 to 39 items"
    )
