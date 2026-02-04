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
SQL Error Detection Data Collator.

Creates error_token_mask for each batch to enable loss computation only on
error token positions (e.g., <no_error>, <error_1>, etc.).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Set

import torch
from transformers import DataCollatorForSeq2Seq

from ...extras.constants import IGNORE_INDEX


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...data.template import Template


@dataclass
class SQLErrorDetectionDataCollator(DataCollatorForSeq2Seq):
    """
    Data collator for SQL Error Detection that creates error_token_mask.
    
    The error_token_mask indicates positions where error tokens appear in the output,
    allowing the trainer to compute loss only on these positions.
    
    Attributes:
        error_token_ids: Set of token IDs that are error tokens (including <no_error>)
        prompt_template: Template used for processing
    """
    
    error_token_ids: Optional[Set[int]] = None
    prompt_lengths: Optional[dict] = None
    processor: Optional["ProcessorMixin"] = None  # Accept processor from tokenizer_module
    
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        """
        Collate features and create error_token_mask.
        
        Args:
            features: List of feature dictionaries containing input_ids, attention_mask, labels
        
        Returns:
            Batch dictionary with added error_token_mask
        """
        # Remove multimodal fields that may contain None values
        # These are added by LlamaFactory's data processing even for text-only data
        for feature in features:
            feature.pop("images", None)
            feature.pop("videos", None)
            feature.pop("audios", None)
        
        # First, use parent class to handle standard collation
        batch = super().__call__(features)
        
        # Create error_token_mask if error_token_ids are provided
        if self.error_token_ids is not None:
            error_token_mask = self._create_error_token_mask(
                batch["input_ids"],
                batch.get("labels"),
            )
            batch["error_token_mask"] = error_token_mask
        
        return batch
    
    def _create_error_token_mask(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Create a mask where 1 = error token position, 0 = other positions.
        
        The mask is based on labels (target tokens) since we want to compute loss
        only when predicting error tokens in the output.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Label token IDs [batch_size, seq_len], may contain IGNORE_INDEX
        
        Returns:
            error_token_mask: Binary mask [batch_size, seq_len]
        """
        if labels is None:
            # If no labels, create mask from input_ids (inference mode)
            target = input_ids
        else:
            target = labels
        
        # Initialize mask with zeros
        mask = torch.zeros_like(target, dtype=torch.float)
        
        # Set mask to 1 for error token positions
        for token_id in self.error_token_ids:
            mask = mask + (target == token_id).float()
        
        # Ensure mask is binary (in case of overlaps)
        mask = (mask > 0).float()
        
        return mask


@dataclass
class SQLErrorDetectionDataCollatorWith4DAttentionMask(SQLErrorDetectionDataCollator):
    """
    Extended collator that also supports 4D attention masks for packed sequences.
    """
    
    template: Optional["Template"] = None
    block_diag_attn: bool = False
    attn_implementation: str = "eager"
    compute_dtype: torch.dtype = torch.float32
    
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        """Collate with 4D attention mask support."""
        # Call parent to get batch with error_token_mask
        batch = super().__call__(features)
        
        # Handle 4D attention mask if needed
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            batch["attention_mask"] = self._prepare_4d_attention_mask(
                batch["attention_mask"], self.compute_dtype
            )
        
        # Cast data dtype
        for key, value in batch.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                batch[key] = value.to(self.compute_dtype)
        
        return batch
    
    def _prepare_4d_attention_mask(
        self,
        attention_mask_with_indices: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Expand 2d attention mask to 4d attention mask.
        
        Handle packed sequences and transforms the mask to lower triangular form.
        """
        _, seq_len = attention_mask_with_indices.size()
        min_dtype = torch.finfo(dtype).min
        zero_tensor = torch.tensor(0, dtype=dtype)
        
        # Create a non-padding mask
        non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
        
        # Create indices for comparison
        indices = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)
        indices_t = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)
        
        # Create a lower triangular mask
        tril_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
        attention_mask_4d = (indices == indices_t) & non_padding_mask & tril_mask
        
        # Invert the attention mask
        attention_mask_4d = torch.where(attention_mask_4d, zero_tensor, min_dtype)
        
        return attention_mask_4d


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
        if token_id != tokenizer.unk_token_id:  # Valid token
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
