from transformers.models.whisper.feature_extraction_whisper import WhisperFeatureExtractor

class WhisperWAFeatureExtractor(WhisperFeatureExtractor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)