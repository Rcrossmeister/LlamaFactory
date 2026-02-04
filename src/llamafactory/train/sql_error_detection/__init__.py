# Copyright 2025 the LlamaFactory team.
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

"""SQL Error Detection Training Module."""

from .workflow import run_sql_error_detection
from .trainer import SQLErrorDetectionTrainer
from .collator import SQLErrorDetectionDataCollator
from .constrained_decoding import (
    ErrorTokenConstraint,
    ErrorTokenOnlyConstraint,
    get_error_token_logits_processor,
)

__all__ = [
    "run_sql_error_detection",
    "SQLErrorDetectionTrainer",
    "SQLErrorDetectionDataCollator",
    "ErrorTokenConstraint",
    "ErrorTokenOnlyConstraint",
    "get_error_token_logits_processor",
]
