from transformers import WhisperConfig

class WhisperWAConfig(WhisperConfig):
    model_type = "whisper_wa"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "whisper_wa"