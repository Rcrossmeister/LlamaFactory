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
Constrained Decoding for AST-Level SQL Error Detection.

This module implements LogitsProcessors that constrain the model to only
generate valid error tokens during inference. Unlike node-level detection,
AST-level detection outputs ONLY error tokens (no "Node[X]: " prefixes).

Example valid outputs:
- <no_error>
- <error_7><error_11><error_29>
- <error_1><error_24>

This is achieved by masking all non-error-token logits to -inf.
"""

from typing import TYPE_CHECKING, Optional, Set

import torch
from transformers import LogitsProcessor, LogitsProcessorList


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class ASTErrorTokenConstraint(LogitsProcessor):
    """
    LogitsProcessor that only allows error tokens throughout the entire output.
    
    For AST-level detection, the model should ONLY generate error tokens
    (e.g., <no_error>, <error_1>, ..., <error_31>) and EOS token.
    
    This constraint:
    1. Allows all error tokens at every position
    2. Allows EOS token to end generation
    3. Blocks all other tokens (regular text, formatting, etc.)
    
    Args:
        error_token_ids: Set of valid error token IDs
        eos_token_id: EOS token ID for ending generation
        allow_special_tokens: Optional set of additional allowed token IDs
    """
    
    def __init__(
        self,
        error_token_ids: Set[int],
        eos_token_id: Optional[int] = None,
        allow_special_tokens: Optional[Set[int]] = None,
    ):
        self.error_token_ids = set(error_token_ids)
        self.eos_token_id = eos_token_id
        self.allow_special_tokens = allow_special_tokens or set()
        
        # Build the set of allowed token IDs
        self.allowed_ids = self.error_token_ids | self.allow_special_tokens
        if eos_token_id is not None:
            self.allowed_ids.add(eos_token_id)
        
        # Cache for the mask tensor
        self._mask_cache = {}
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Process logits to only allow error tokens and EOS.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            scores: Logits for next token [batch_size, vocab_size]
        
        Returns:
            Modified scores with non-error tokens masked to -inf
        """
        vocab_size = scores.shape[1]
        device = scores.device
        
        # Use cached mask if available for this vocab_size and device
        cache_key = (vocab_size, device)
        if cache_key not in self._mask_cache:
            # Create mask: -inf for disallowed tokens, 0 for allowed
            mask = torch.full((vocab_size,), float('-inf'), device=device)
            for token_id in self.allowed_ids:
                if token_id < vocab_size:
                    mask[token_id] = 0.0
            self._mask_cache[cache_key] = mask
        
        mask = self._mask_cache[cache_key]
        
        # Apply mask to all batch items
        scores = scores + mask.unsqueeze(0)
        
        return scores


class ASTErrorTokenSequenceConstraint(LogitsProcessor):
    """
    More sophisticated constraint that enforces valid error token sequences.
    
    Rules:
    1. If <no_error> is generated, it must be the only token (followed by EOS)
    2. Error tokens should be in ascending order (e.g., <error_1><error_7>, not <error_7><error_1>)
    3. No duplicate error tokens
    
    Args:
        tokenizer: Tokenizer for decoding
        error_token_ids: Set of valid error token IDs
        no_error_id: Token ID for <no_error>
        eos_token_id: EOS token ID
    """
    
    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        error_token_ids: Set[int],
        no_error_id: int,
        eos_token_id: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.error_token_ids = set(error_token_ids)
        self.no_error_id = no_error_id
        self.eos_token_id = eos_token_id
        
        # Build error token to number mapping for ordering
        self.error_token_to_num = {}
        for token_id in error_token_ids:
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token == "<no_error>":
                self.error_token_to_num[token_id] = -1  # Special case
            elif token.startswith("<error_"):
                try:
                    num = int(token[7:-1])  # Extract number from <error_X>
                    self.error_token_to_num[token_id] = num
                except (ValueError, IndexError):
                    self.error_token_to_num[token_id] = 999  # Unknown
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Process logits with sequence constraints.
        """
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[1]
        
        for batch_idx in range(batch_size):
            # Find generated error tokens in this sequence
            generated_ids = input_ids[batch_idx].tolist()
            generated_error_tokens = []
            generated_error_nums = []
            
            for token_id in generated_ids:
                if token_id in self.error_token_ids:
                    generated_error_tokens.append(token_id)
                    if token_id in self.error_token_to_num:
                        generated_error_nums.append(self.error_token_to_num[token_id])
            
            # Apply constraints
            allowed_ids = set()
            
            # If <no_error> was generated, only allow EOS
            if self.no_error_id in generated_error_tokens:
                if self.eos_token_id is not None:
                    allowed_ids.add(self.eos_token_id)
            else:
                # Allow EOS
                if self.eos_token_id is not None:
                    allowed_ids.add(self.eos_token_id)
                
                # Allow error tokens that haven't been generated yet
                # and have higher numbers than the last generated
                max_num = max(generated_error_nums) if generated_error_nums else -1
                
                for token_id in self.error_token_ids:
                    if token_id == self.no_error_id:
                        # Only allow <no_error> if nothing else has been generated
                        if not generated_error_tokens:
                            allowed_ids.add(token_id)
                    elif token_id not in generated_error_tokens:
                        # Allow if this error number is greater than max generated
                        token_num = self.error_token_to_num.get(token_id, 999)
                        if token_num > max_num:
                            allowed_ids.add(token_id)
            
            # Create mask for this batch item
            mask = torch.full((vocab_size,), float('-inf'), device=scores.device)
            for token_id in allowed_ids:
                if token_id < vocab_size:
                    mask[token_id] = 0.0
            
            scores[batch_idx] = scores[batch_idx] + mask
        
        return scores


def get_ast_error_token_logits_processor(
    tokenizer: "PreTrainedTokenizer",
    error_token_ids: Set[int],
    no_error_id: Optional[int] = None,
    constraint_type: str = "simple",
    allow_eos: bool = True,
) -> LogitsProcessorList:
    """
    Create a LogitsProcessorList with appropriate error token constraints for AST-level detection.
    
    Args:
        tokenizer: The tokenizer
        error_token_ids: Set of error token IDs
        no_error_id: Token ID for <no_error> (required for "ordered" constraint)
        constraint_type: Type of constraint:
            - "simple": Only allow error tokens and EOS (recommended for most cases)
            - "ordered": Enforce ascending order and no duplicates
        allow_eos: Whether to allow EOS token (for ending generation)
    
    Returns:
        LogitsProcessorList with the appropriate constraint
    """
    processors = []
    
    eos_token_id = tokenizer.eos_token_id if allow_eos else None
    
    if constraint_type == "simple":
        processors.append(ASTErrorTokenConstraint(
            error_token_ids=error_token_ids,
            eos_token_id=eos_token_id,
        ))
    elif constraint_type == "ordered":
        if no_error_id is None:
            no_error_id = tokenizer.convert_tokens_to_ids("<no_error>")
        processors.append(ASTErrorTokenSequenceConstraint(
            tokenizer=tokenizer,
            error_token_ids=error_token_ids,
            no_error_id=no_error_id,
            eos_token_id=eos_token_id,
        ))
    
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
