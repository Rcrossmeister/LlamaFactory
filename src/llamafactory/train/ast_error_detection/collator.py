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

"""
AST-Level SQL Error Detection Data Collator.

Standard data collator for AST-level detection. The output format is simply
concatenated error tokens (e.g., <error_7><error_11><error_29> or <no_error>).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Set

from transformers import DataCollatorForSeq2Seq


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@dataclass
class ASTErrorDetectionDataCollator(DataCollatorForSeq2Seq):
    """
    Data collator for AST-Level SQL Error Detection.
    
    Uses standard DataCollatorForSeq2Seq behavior - no special masking needed
    since AST-level detection computes loss on all output tokens.
    """
    
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate features.
        
        Args:
            features: List of feature dictionaries
        
        Returns:
            Batch dictionary
        """
        # Remove multimodal fields that may contain None values
        for feature in features:
            feature.pop("images", None)
            feature.pop("videos", None)
            feature.pop("audios", None)
        
        return super().__call__(features)


def get_error_token_ids(tokenizer: "PreTrainedTokenizer") -> Set[int]:
    """
    Get the set of error token IDs from the tokenizer.
    
    Args:
        tokenizer: Tokenizer with error tokens added
    
    Returns:
        Set of token IDs for error tokens
    """
    error_tokens = [
        "<no_error>",
        "<error_1>", "<error_2>", "<error_3>",
        "<error_4>", "<error_5>", "<error_6>", "<error_7>", "<error_8>",
        "<error_9>", "<error_10>",
        "<error_11>", "<error_12>",
        "<error_13>", "<error_14>", "<error_15>", "<error_16>",
        "<error_17>", "<error_18>", "<error_19>", "<error_20>",
        "<error_21>", "<error_22>", "<error_23>",
        "<error_24>", "<error_25>",
        "<error_26>", "<error_27>", "<error_28>",
        "<error_29>", "<error_30>", "<error_31>",
    ]
    
    error_token_ids = set()
    for token in error_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            error_token_ids.add(token_id)
    
    return error_token_ids


def get_no_error_token_id(tokenizer: "PreTrainedTokenizer") -> int:
    """
    Get the token ID for <no_error>.
    
    Args:
        tokenizer: Tokenizer with error tokens added
    
    Returns:
        Token ID for <no_error>
    """
    return tokenizer.convert_tokens_to_ids("<no_error>")
