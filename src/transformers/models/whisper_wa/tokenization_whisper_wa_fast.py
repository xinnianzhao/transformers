from transformers.models.whisper.tokenization_whisper_fast import WhisperTokenizerFast
from ...utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "tokenizer_file": "tokenizer.json",
    "merges_file": "merges.txt",
    "normalizer_file": "normalizer.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "openai/whisper-tiny": "https://huggingface.co/openai/whisper-tiny/resolve/main/vocab.json",
        "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/vocab.json",
        "openai/whisper-small": "https://huggingface.co/openai/whisper-small/resolve/main/vocab.json",
        "openai/whisper-medium": "https://huggingface.co/openai/whisper-medium/resolve/main/vocab.json",
        "openai/whisper-large": "https://huggingface.co/openai/whisper-large/resolve/main/vocab.json",
        "openai/whisper-tiny.en": "https://huggingface.co/openai/whisper-tiny.en/resolve/main/vocab.json",
        "openai/whisper-base.en": "https://huggingface.co/openai/whisper-base.en/resolve/main/vocab.json",
        "openai/whisper-small.en": "https://huggingface.co/openai/whisper-small.en/resolve/main/vocab.json",
        "openai/whisper-medium.en": "https://huggingface.co/openai/whisper-medium.en/resolve/main/vocab.json",
    },
    "merges_file": {
        "openai/whisper-tiny": "https://huggingface.co/openai/whisper-tiny/resolve/main/merges.txt",
        "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/merges.txt",
        "openai/whisper-small": "https://huggingface.co/openai/whisper-small/resolve/main/merges.txt",
        "openai/whisper-medium": "https://huggingface.co/openai/whisper-medium/resolve/main/merges.txt",
        "openai/whisper-large": "https://huggingface.co/openai/whisper-large/resolve/main/merges.txt",
        "openai/whisper-tiny.en": "https://huggingface.co/openai/whisper-tiny.en/resolve/main/merges.txt",
        "openai/whisper-base.en": "https://huggingface.co/openai/whisper-base.en/resolve/main/merges.txt",
        "openai/whisper-small.en": "https://huggingface.co/openai/whisper-small.en/resolve/main/merges.txt",
        "openai/whisper-medium.en": "https://huggingface.co/openai/whisper-medium.en/resolve/main/merges.txt",
    },
    "tokenizer_file": {
        "openai/whisper-tiny": "https://huggingface.co/openai/whisper-tiny/resolve/main/tokenizer.json",
        "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/tokenizer.json",
        "openai/whisper-small": "https://huggingface.co/openai/whisper-small/resolve/main/tokenizer.json",
        "openai/whisper-medium": "https://huggingface.co/openai/whisper-medium/resolve/main/tokenizer.json",
        "openai/whisper-large": "https://huggingface.co/openai/whisper-large/resolve/main/tokenizer.json",
        "openai/whisper-tiny.en": "https://huggingface.co/openai/whisper-tiny.en/resolve/main/tokenizer.json",
        "openai/whisper-base.en": "https://huggingface.co/openai/whisper-base.en/resolve/main/tokenizer.json",
        "openai/whisper-small.en": "https://huggingface.co/openai/whisper-small.en/resolve/main/tokenizer.json",
        "openai/whisper-medium.en": "https://huggingface.co/openai/whisper-medium.en/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "openai/whisper-tiny": 1500,
    "openai/whisper-base": 1500,
    "openai/whisper-small": 1500,
    "openai/whisper-medium": 1500,
    "openai/whisper-large": 1500,
    "openai/whisper-tiny.en": 1500,
    "openai/whisper-base.en": 1500,
    "openai/whisper-small.en": 1500,
    "openai/whisper-medium.en": 1500,
}


class WhisperWATokenizerFast(WhisperTokenizerFast):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)