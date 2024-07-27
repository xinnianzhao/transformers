# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import TYPE_CHECKING

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_whisper_wa": ["WhisperWAConfig"],
    "feature_extraction_whisper_wa": ["WhisperWAFeatureExtractor"],
    "processing_whisper_wa": ["WhisperWAProcessor"],
    "tokenization_whisper_wa": ["WhisperWATokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_whisper_wa_fast"] = ["WhisperWATokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_whisper_wa"] = [
        "WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST",
        "WhisperWAForCausalLM",
        "WhisperWAForConditionalGeneration",
        "WhisperWAModel",
        "WhisperPreTrainedModel",
        "WhisperWAForAudioClassification",
    ]


if TYPE_CHECKING:
    from .configuration_whisper_wa import WhisperWAConfig
    from .feature_extraction_whisper_wa import WhisperWAFeatureExtractor
    from .processing_whisper_wa import WhisperWAProcessor
    from .tokenization_whisper_wa import WhisperWATokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_whisper_wa_fast import WhisperWATokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_whisper_wa import (
            WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST,
            WhisperWAForAudioClassification,
            WhisperWAForCausalLM,
            WhisperWAForConditionalGeneration,
            WhisperWAModel,
            WhisperPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
