"""
constants.py
"""

import os

from pathlib import Path

# Key constants
APP_TITLE = "Azure NotebookLM"
CHARACTER_LIMIT = 100_000

# Gradio-related constants
GRADIO_CACHE_DIR = "./gradio_cached_examples/tmp/"
GRADIO_CLEAR_CACHE_OLDER_THAN = 1 * 24 * 60 * 60  # 1 day

# Error messages-related constants
ERROR_MESSAGE_NO_INPUT = "Please provide at least one input: upload PDF file(s), enter a URL, or paste text directly."
ERROR_MESSAGE_NOT_PDF = "The provided file is not a PDF. Please upload only PDF files."
ERROR_MESSAGE_NOT_SUPPORTED_IN_MELO_TTS = "The selected language is not supported without advanced audio generation. Please enable advanced audio generation or choose a supported language."
ERROR_MESSAGE_READING_PDF = "Error reading the PDF file"
ERROR_MESSAGE_TOO_LONG = "The total content is too long. Please ensure the combined text from PDFs, URL, and direct text input is fewer than {CHARACTER_LIMIT} characters."

JSON_RETRY_ATTEMPTS = 1

# Jina Reader-related constants
JINA_READER_URL = "https://r.jina.ai/"
JINA_RETRY_ATTEMPTS = 3
JINA_RETRY_DELAY = 5  # in seconds

# Speech synthesis constants
SPEECH_RATE = "1.1"  # 1.0 = normal speed, 1.2 = 20% faster, 0.8 = 20% slower
SPEECH_FORMAT = "Riff48Khz16BitMonoPcm"  # Highest quality uncompressed 48kHz 16-bit WAV format
AUDIO_FILE_EXTENSION = ".wav"

# Suno related constants
LANGUAGE_MAPPING = {
    "English": "en",
    "German": "de",
    "French": "fr"
}

# UI-related constants
UI_DESCRIPTION = """
<table style="border-collapse: collapse; border: none; padding: 20px;">
  <tr style="border: none;">
    <td style="border: none; vertical-align: top; padding-right: 30px; padding-left: 30px;">
      <img src="https://upload.wikimedia.org/wikipedia/commons/f/fa/Microsoft_Azure.svg" alt="Azure" width="120" style="margin-bottom: 10px;">
    </td>
    <td style="border: none; vertical-align: top; padding: 10px;">
      <p style="margin-bottom: 15px;">Convert your PDFs into podcasts with Azure AI serivces (<a href="https://azure.microsoft.com/en-us/products/ai-services/openai-service">Azure OpenAI</a>, <a href="https://azure.microsoft.com/en-us/products/ai-services/ai-speech">Azure AI Speech </a>).</p>
      <p style="margin-top: 15px;">Note: Only the text content of the PDFs will be processed. Images and tables are not included. The total content should be no more than 100,000 characters.</p>
    </td>
  </tr>
</table>
"""
UI_AVAILABLE_LANGUAGES = list(set(LANGUAGE_MAPPING.keys()))
UI_INPUTS = {
    "file_upload": {
        "label": "1. 📄 Upload your PDF(s)",
        "file_types": [".pdf"],
        "file_count": "multiple",
    },
    "url": {
        "label": "2. 🔗 Paste a URL (optional)",
        "placeholder": "Enter a URL to include its content",
    },
    "text_input": {
        "label": "3. 📝 Or paste your text directly (optional)",
        "placeholder": "Paste your text content here as an alternative to PDFs or URLs",
        "lines": 5,
    },
    "question": {
        "label": "4. 🤔 Do you have a specific question or topic in mind?",
        "placeholder": "Enter a question or topic",
    },
    "tone": {
        "label": "5. 🎭 Choose the tone",
        "choices": ["Fun", "Formal"],
        "value": "Fun",
    },
    "length": {
        "label": "6. ⏱️ Choose the length",
        "choices": ["Short (1-2 min)", "Medium (3-5 min)"],
        "value": "Short (1-2 min)",
    },
    "language": {
        "label": "7. 🌐 Choose the language",
        "choices": UI_AVAILABLE_LANGUAGES,
        "value": "English",
    },
}
UI_OUTPUTS = {
    "audio": {"label": "🔊 Podcast", "format": "wav"},
    "transcript": {
        "label": "📜 Transcript",
    },
}
UI_API_NAME = "generate_podcast"
UI_ALLOW_FLAGGING = "never"
UI_CONCURRENCY_LIMIT = 3
UI_EXAMPLES = [
    [
        [str(Path("examples/KFE_news_sample.pdf"))],
        "",
        "",
        "",
        "Fun",
        "Short (1-2 min)",
        "English",
        True,
    ],
    [
        [str(Path("examples/KFE_paper_sample.pdf"))],
        "",
        "",
        "",
        "Fun",
        "Medium (3-5 min)",
        "English",
        True,
    ],
]
UI_CACHE_EXAMPLES = True
UI_SHOW_API = True
