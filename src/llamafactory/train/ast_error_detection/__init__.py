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

"""AST-Level SQL Error Detection Training Module.

This module provides AST-level (query-level) SQL error detection, where
the model outputs concatenated error tokens representing all errors found
in a SQL query.

Example outputs:
- <error_7><error_11><error_29>  (multiple errors)
- <no_error>                    (no errors)

Key features:
- Extended vocabulary with 32 error tokens
- Embedding freeze hooks (train new tokens, freeze original)
- Constrained decoding during inference
- Standard cross-entropy loss (or optional focal loss)
"""

from .workflow import run_ast_error_detection
from .trainer import (
    ASTErrorDetectionTrainer,
    ERROR_TOKENS,
    setup_embedding_freeze_hooks,
    initialize_error_token_embeddings,
)
from .collator import (
    ASTErrorDetectionDataCollator,
    get_error_token_ids,
    get_no_error_token_id,
)
from .constrained_decoding import (
    ASTErrorTokenConstraint,
    get_ast_error_token_logits_processor,
)

__all__ = [
    "run_ast_error_detection",
    "ASTErrorDetectionTrainer",
    "ERROR_TOKENS",
    "setup_embedding_freeze_hooks",
    "initialize_error_token_embeddings",
    "ASTErrorDetectionDataCollator",
    "get_error_token_ids",
    "get_no_error_token_id",
    "ASTErrorTokenConstraint",
    "get_ast_error_token_logits_processor",
]
