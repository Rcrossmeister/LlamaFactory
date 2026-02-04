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
SQL Error Detection Trainer with Focal Loss.

This trainer implements:
1. Focal Loss for class imbalance (error tokens are much rarer than <no_error>)
2. Error token masking (only compute loss on error token positions)
3. Support for freezing original embeddings while training new error token embeddings
"""

import json
import os
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.get_logger(__name__)


# Error tokens for SQL error detection (32 tokens total)
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


class FocalLoss:
    """
    Focal Loss for handling class imbalance in SQL error detection.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focusing parameter that increases focus on hard samples. Default: 2.0
        alpha: Class weights tensor of shape [vocab_size]. If None, no class weighting.
        no_error_weight: Weight for <no_error> token. Default: 0.1 (down-weight dominant class)
        error_token_weight: Weight for error tokens <error_1> to <error_31>. Default: 1.0
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        no_error_weight: float = 0.1,
        error_token_ids: Optional[set] = None,
        no_error_id: Optional[int] = None,
        vocab_size: Optional[int] = None,
        error_token_weight: float = 1.0,
    ):
        self.gamma = gamma
        self.no_error_weight = no_error_weight
        self.error_token_ids = error_token_ids or set()
        self.no_error_id = no_error_id
        self.error_token_weight = error_token_weight
        
        if alpha is not None:
            self.alpha = alpha
        elif vocab_size is not None and error_token_ids is not None:
            # Create class weights - ONLY for error tokens
            # Regular text tokens will have weight 1.0 (neutral)
            # <no_error> tokens will be down-weighted
            # Other error tokens will have normal or up-weighted
            self.alpha = torch.ones(vocab_size)
            
            # Down-weight <no_error> (dominant class)
            if no_error_id is not None:
                self.alpha[no_error_id] = no_error_weight
            
            # Set weight for other error tokens (can be up-weighted to focus more on errors)
            for token_id in error_token_ids:
                if token_id != no_error_id:
                    self.alpha[token_id] = error_token_weight
        else:
            self.alpha = None
    
    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size] or [N, vocab_size]
            labels: Ground truth labels [batch_size, seq_len] or [N]
            mask: Optional mask where 1 = compute loss, 0 = ignore [batch_size, seq_len] or [N]
        
        Returns:
            Scalar loss value
        """
        # Flatten if needed
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.size()
            logits = logits.view(-1, vocab_size)
            labels = labels.view(-1)
            if mask is not None:
                mask = mask.view(-1)
        
        # Create valid mask for loss computation (exclude IGNORE_INDEX)
        valid_mask = labels != IGNORE_INDEX
        
        # Compute cross entropy loss (per element) only on valid positions
        # Use ignore_index to avoid computing loss on padding
        ce_loss = F.cross_entropy(logits, labels, reduction='none', ignore_index=IGNORE_INDEX)
        
        # Compute pt (probability of true class)
        pt = torch.exp(-ce_loss)
        
        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights ONLY to error token positions
        # This is the key fix - we don't apply alpha to regular text tokens
        if self.alpha is not None and self.error_token_ids:
            alpha = self.alpha.to(logits.device)
            
            # Create a mask for positions where label is an error token
            error_token_mask = torch.zeros_like(labels, dtype=torch.bool)
            for token_id in self.error_token_ids:
                error_token_mask = error_token_mask | (labels == token_id)
            
            # Apply alpha weights only to error token positions
            # Clamp labels to valid range before indexing (handle IGNORE_INDEX=-100)
            safe_labels = labels.clamp(min=0)
            alpha_t = alpha[safe_labels]
            
            # Only apply alpha to error token positions, others keep weight 1.0
            alpha_weights = torch.where(error_token_mask, alpha_t, torch.ones_like(alpha_t))
            focal_loss = alpha_weights * focal_loss
        
        # Apply mask if provided
        if mask is not None:
            # Only compute loss on masked positions
            valid_mask = (labels != IGNORE_INDEX) & (mask > 0)
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            return (focal_loss * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)
        else:
            # Standard case - ignore IGNORE_INDEX positions
            valid_mask = labels != IGNORE_INDEX
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            return (focal_loss * valid_mask.float()).sum() / (valid_mask.sum() + 1e-8)


class SQLErrorDetectionTrainer(Seq2SeqTrainer):
    """
    Custom trainer for SQL Error Detection with:
    1. Focal Loss for class imbalance
    2. Error token position masking
    3. Support for new token embedding training
    """
    
    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        error_token_ids: Optional[set] = None,
        no_error_id: Optional[int] = None,
        original_vocab_size: Optional[int] = None,
        focal_gamma: float = 2.0,
        no_error_weight: float = 0.1,
        error_token_weight: float = 1.0,
        use_error_token_mask: bool = True,
        **kwargs,
    ) -> None:
        """
        Initialize SQL Error Detection Trainer.
        
        Args:
            finetuning_args: Finetuning arguments
            processor: Optional processor
            model_args: Model arguments
            gen_kwargs: Generation kwargs
            error_token_ids: Set of token IDs for error tokens (including <no_error>)
            no_error_id: Token ID for <no_error>
            original_vocab_size: Original vocabulary size before adding error tokens
            focal_gamma: Focal loss gamma parameter
            no_error_weight: Weight for <no_error> class in focal loss
            error_token_weight: Weight for error tokens <error_1> to <error_31> (default: 1.0)
            use_error_token_mask: Whether to only compute loss on error token positions
        """
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(**kwargs)
        
        if processor is not None:
            self.model_accepts_loss_kwargs = False
        
        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            self._gen_kwargs = gen_kwargs
        
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))
        
        # SQL Error Detection specific settings
        self.error_token_ids = error_token_ids or set()
        self.no_error_id = no_error_id
        self.original_vocab_size = original_vocab_size
        self.use_error_token_mask = use_error_token_mask
        
        # Initialize Focal Loss
        vocab_size = self.model.config.vocab_size if hasattr(self.model, 'config') else None
        self.focal_loss = FocalLoss(
            gamma=focal_gamma,
            no_error_weight=no_error_weight,
            error_token_ids=error_token_ids,
            no_error_id=no_error_id,
            vocab_size=vocab_size,
            error_token_weight=error_token_weight,
        )
        
        logger.info_rank0(f"SQL Error Detection Trainer initialized:")
        logger.info_rank0(f"  - Focal gamma: {focal_gamma}")
        logger.info_rank0(f"  - No error weight: {no_error_weight}")
        logger.info_rank0(f"  - Error token weight: {error_token_weight}")
        logger.info_rank0(f"  - Use error token mask: {use_error_token_mask}")
        logger.info_rank0(f"  - Error token IDs: {len(self.error_token_ids)} tokens")
    
    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()
    
    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)
    
    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)
    
    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute focal loss with optional error token masking.
        
        This overrides the default cross-entropy loss with:
        1. Focal loss for better handling of class imbalance
        2. Optional masking to only compute loss on error token positions
        """
        # Get error_token_mask from inputs if available
        error_token_mask = inputs.pop("error_token_mask", None)
        
        # Forward pass
        outputs = model(**inputs)
        
        if hasattr(outputs, "loss") and outputs.loss is not None:
            # If model computes its own loss, we still override it with focal loss
            pass
        
        # Get logits and labels
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        labels = inputs.get("labels")
        
        if labels is None:
            # Fallback to standard loss if no labels
            if return_outputs:
                return outputs.loss, outputs
            return outputs.loss
        
        # Shift for causal LM (predict next token)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Shift error token mask if provided
        if error_token_mask is not None and self.use_error_token_mask:
            shift_mask = error_token_mask[..., 1:].contiguous()
        else:
            shift_mask = None
        
        # Compute focal loss
        loss = self.focal_loss(shift_logits, shift_labels, shift_mask)
        
        if return_outputs:
            outputs.loss = loss
            return loss, outputs
        
        return loss
    
    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        """Remove the prompt part in the generated tokens."""
        # Remove error_token_mask from inputs for prediction
        inputs.pop("error_token_mask", None)
        
        if self.args.predict_with_generate:
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")
        
        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()
        
        return loss, generated_tokens, labels
    
    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        """Save model predictions to `output_dir`."""
        if not self.is_world_process_zero():
            return
        
        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")
        
        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )
        
        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate((preds[i][pad_len[0]:], preds[i][:pad_len[0]]), axis=-1)
        
        decoded_inputs = [self.processing_class.decode(ids, skip_special_tokens=False) for ids in dataset["input_ids"]]
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)
        
        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")


def setup_embedding_freeze_hooks(
    model: torch.nn.Module,
    original_vocab_size: int,
    device: Optional[torch.device] = None,
) -> None:
    """
    Set up gradient hooks to freeze original embeddings while allowing new embeddings to train.
    
    This ensures that:
    1. Original vocabulary embeddings remain frozen
    2. New error token embeddings can be trained
    
    Args:
        model: The model to set up hooks for
        original_vocab_size: Size of original vocabulary (before adding error tokens)
        device: Device to use for mask tensor
    """
    def create_freeze_hook(vocab_size: int):
        """Create a hook that zeros gradients for original vocabulary."""
        def freeze_hook(grad):
            if grad is None:
                return grad
            # Zero out gradients for original vocabulary
            grad[:vocab_size] = 0
            return grad
        return freeze_hook
    
    # Get embedding layer
    embed_layer = model.get_input_embeddings()
    if embed_layer is not None and hasattr(embed_layer, 'weight'):
        embed_layer.weight.register_hook(create_freeze_hook(original_vocab_size))
        logger.info_rank0(f"Registered freeze hook for input embeddings (original vocab size: {original_vocab_size})")
    
    # Get LM head layer (if not tied)
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        if hasattr(model.lm_head, 'weight'):
            # Check if weights are tied
            embed_weight = embed_layer.weight if embed_layer is not None else None
            lm_head_weight = model.lm_head.weight
            
            if embed_weight is None or not torch.equal(embed_weight.data, lm_head_weight.data):
                model.lm_head.weight.register_hook(create_freeze_hook(original_vocab_size))
                logger.info_rank0(f"Registered freeze hook for LM head (original vocab size: {original_vocab_size})")
            else:
                logger.info_rank0("LM head weights are tied to embeddings, skipping separate hook")


def initialize_error_token_embeddings(
    model: torch.nn.Module,
    tokenizer,
    error_tokens: list[str],
    original_vocab_size: int,
) -> None:
    """
    Initialize new error token embeddings using semantic averaging.
    
    Each error token embedding is initialized as the average of embeddings
    from semantically related words.
    
    Args:
        model: The model with resized embeddings
        tokenizer: Tokenizer with added error tokens
        error_tokens: List of error token strings
        original_vocab_size: Original vocabulary size before adding tokens
    """
    # Semantic descriptions for error tokens
    ERROR_TOKEN_SEMANTICS = {
        "<no_error>": "correct valid no error",
        "<error_1>": "attribute mismatch wrong column",
        "<error_2>": "attribute redundancy extra column",
        "<error_3>": "attribute missing absent column",
        "<error_4>": "table mismatch wrong table",
        "<error_5>": "table redundancy extra table",
        "<error_6>": "table missing absent",
        "<error_7>": "join condition mismatch",
        "<error_8>": "join type mismatch",
        "<error_9>": "value mismatch wrong literal",
        "<error_10>": "data format mismatch type",
        "<error_11>": "comparison operator error",
        "<error_12>": "logical operator error",
        "<error_13>": "explicit condition missing",
        "<error_14>": "explicit condition mismatch",
        "<error_15>": "explicit condition redundancy",
        "<error_16>": "implicit condition missing",
        "<error_17>": "aggregate function error",
        "<error_18>": "window function error",
        "<error_19>": "datetime function error",
        "<error_20>": "conversion function error",
        "<error_21>": "math function error",
        "<error_22>": "string function error",
        "<error_23>": "conditional function error",
        "<error_24>": "clause missing",
        "<error_25>": "clause redundancy",
        "<error_26>": "subquery missing",
        "<error_27>": "subquery mismatch",
        "<error_28>": "partial query incomplete",
        "<error_29>": "ascending descending order error",
        "<error_30>": "distinct error duplicate",
        "<error_31>": "other error unknown",
    }
    
    embed_weight = model.get_input_embeddings().weight
    
    with torch.no_grad():
        for token in error_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            
            # Skip if token is in original vocabulary (shouldn't happen)
            if token_id < original_vocab_size:
                continue
            
            # Get semantic description
            semantics = ERROR_TOKEN_SEMANTICS.get(token, "error unknown")
            
            # Collect embeddings of semantic words
            word_embeds = []
            for word in semantics.split():
                word_ids = tokenizer.encode(word, add_special_tokens=False)
                for wid in word_ids:
                    if wid < original_vocab_size:  # Only use original vocab embeddings
                        word_embeds.append(embed_weight[wid].clone())
            
            # Average embeddings
            if word_embeds:
                embed_weight[token_id] = torch.stack(word_embeds).mean(dim=0)
                logger.info_rank0(f"Initialized {token} embedding from {len(word_embeds)} semantic tokens")
            else:
                logger.warning_rank0(f"No semantic tokens found for {token}, using random initialization")
