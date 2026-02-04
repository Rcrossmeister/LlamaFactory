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
Constrained Decoding for SQL Error Detection.

This module implements LogitsProcessors that constrain the model to only
generate valid error tokens at specific positions during inference.
"""

import re
from typing import TYPE_CHECKING, Optional, Set

import torch
from transformers import LogitsProcessor, LogitsProcessorList


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class ErrorTokenConstraint(LogitsProcessor):
    """
    LogitsProcessor that forces the model to generate error tokens after "Node[X]: " patterns.
    
    This ensures the model outputs valid error tokens (<no_error>, <error_1>, etc.)
    at the correct positions, preventing invalid outputs like:
    - "Node[1]: no error" (missing angle brackets)
    - "Node[1]: <error_" (incomplete token)
    - "Node[1]: The error is..." (free text)
    
    Args:
        tokenizer: The tokenizer used for decoding
        error_token_ids: Set of valid error token IDs
        context_length: Number of recent tokens to decode for pattern matching
    """
    
    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        error_token_ids: Set[int],
        context_length: int = 20,
    ):
        self.tokenizer = tokenizer
        self.error_token_ids = set(error_token_ids)
        self.context_length = context_length
        # Pattern to match "Node[X]: " where X is a number
        self.node_pattern = re.compile(r'Node\[\d+\]:\s*$')
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Process logits to constrain generation to error tokens when appropriate.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            scores: Logits for next token [batch_size, vocab_size]
        
        Returns:
            Modified scores with constraints applied
        """
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            # Get recent tokens for this batch item
            recent_ids = input_ids[batch_idx, -self.context_length:]
            recent_text = self.tokenizer.decode(recent_ids, skip_special_tokens=False)
            
            # Check if we just generated "Node[X]: " pattern
            if self.node_pattern.search(recent_text):
                # Force generation of an error token
                # Set all non-error-token scores to -inf
                mask = torch.full_like(scores[batch_idx], float('-inf'))
                for token_id in self.error_token_ids:
                    if token_id < mask.shape[0]:
                        mask[token_id] = 0.0
                scores[batch_idx] = scores[batch_idx] + mask
        
        return scores


class ErrorTokenOnlyConstraint(LogitsProcessor):
    """
    Stricter constraint that only allows error tokens throughout the entire output.
    
    Use this when you want to generate only error tokens (e.g., for structured output
    where the format tokens are handled separately).
    
    Args:
        error_token_ids: Set of valid error token IDs
        allow_special_tokens: Set of additional allowed token IDs (e.g., EOS, comma)
    """
    
    def __init__(
        self,
        error_token_ids: Set[int],
        allow_special_tokens: Optional[Set[int]] = None,
    ):
        self.error_token_ids = set(error_token_ids)
        self.allow_special_tokens = allow_special_tokens or set()
        self.allowed_ids = self.error_token_ids | self.allow_special_tokens
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply constraint to only allow error tokens and special tokens."""
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[1]
        
        # Create mask: -inf for disallowed tokens, 0 for allowed
        mask = torch.full((vocab_size,), float('-inf'), device=scores.device)
        for token_id in self.allowed_ids:
            if token_id < vocab_size:
                mask[token_id] = 0.0
        
        # Apply mask to all batch items
        scores = scores + mask.unsqueeze(0)
        
        return scores


def get_error_token_logits_processor(
    tokenizer: "PreTrainedTokenizer",
    error_token_ids: Set[int],
    constraint_type: str = "node_pattern",
    allow_eos: bool = True,
) -> LogitsProcessorList:
    """
    Create a LogitsProcessorList with appropriate error token constraints.
    
    Args:
        tokenizer: The tokenizer
        error_token_ids: Set of error token IDs
        constraint_type: Type of constraint:
            - "node_pattern": Only constrain after "Node[X]: " patterns
            - "strict": Only allow error tokens throughout
        allow_eos: Whether to allow EOS token (for ending generation)
    
    Returns:
        LogitsProcessorList with the appropriate constraint
    """
    processors = []
    
    if constraint_type == "node_pattern":
        processors.append(ErrorTokenConstraint(tokenizer, error_token_ids))
    elif constraint_type == "strict":
        allow_special = set()
        if allow_eos and tokenizer.eos_token_id is not None:
            allow_special.add(tokenizer.eos_token_id)
        processors.append(ErrorTokenOnlyConstraint(error_token_ids, allow_special))
    
    return LogitsProcessorList(processors)


# Error tokens list for reference
ERROR_TOKENS = [
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


def get_error_token_ids_from_tokenizer(tokenizer: "PreTrainedTokenizer") -> Set[int]:
    """
    Get the set of error token IDs from the tokenizer.
    
    Args:
        tokenizer: Tokenizer with error tokens added
    
    Returns:
        Set of token IDs for error tokens
    """
    error_token_ids = set()
    for token in ERROR_TOKENS:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            error_token_ids.add(token_id)
    return error_token_ids
