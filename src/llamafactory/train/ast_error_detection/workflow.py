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
AST-Level SQL Error Detection Training Workflow.

This module provides the main entry point for AST-level SQL error detection
training, which outputs concatenated error tokens per query (e.g., <error_7><error_11>).

Key features:
1. Loading model with extended vocabulary (error tokens)
2. Embedding freeze hooks for original vocabulary
3. Constrained decoding for inference (only allow error tokens)
4. Standard cross-entropy loss (or optional focal loss)
"""

from typing import TYPE_CHECKING, Optional

from ...data import get_dataset, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.packages import is_transformers_version_greater_than
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..trainer_utils import create_modelcard_and_push
from ..sft.metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
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
from .constrained_decoding import get_ast_error_token_logits_processor


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def run_ast_error_detection(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
    # AST Error Detection specific arguments
    focal_gamma: float = 0.0,  # Default 0 means standard CE loss
    freeze_original_embeddings: bool = True,
    initialize_embeddings: bool = False,
    use_constrained_decoding: bool = True,
    constraint_type: str = "simple",
):
    """
    Run AST-Level SQL Error Detection training.
    
    Args:
        model_args: Model arguments
        data_args: Data arguments
        training_args: Training arguments
        finetuning_args: Finetuning arguments
        generating_args: Generating arguments
        callbacks: Optional trainer callbacks
        focal_gamma: Focal loss gamma (0 = standard CE loss, >0 = focal loss)
        freeze_original_embeddings: Whether to freeze original vocabulary embeddings
        initialize_embeddings: Whether to initialize error token embeddings semantically
        use_constrained_decoding: Whether to use constrained decoding during inference
        constraint_type: Type of constraint ("simple" or "ordered")
    """
    logger.info_rank0("=" * 60)
    logger.info_rank0("Starting AST-Level SQL Error Detection Training")
    logger.info_rank0("=" * 60)
    
    # Load tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    
    # Get error token IDs from tokenizer
    error_token_ids = get_error_token_ids(tokenizer)
    no_error_id = get_no_error_token_id(tokenizer)
    
    logger.info_rank0(f"Found {len(error_token_ids)} error tokens in vocabulary")
    logger.info_rank0(f"<no_error> token ID: {no_error_id}")
    
    # Verify error tokens are in vocabulary
    if len(error_token_ids) != len(ERROR_TOKENS):
        logger.warning_rank0(
            f"Expected {len(ERROR_TOKENS)} error tokens, but found {len(error_token_ids)}. "
            "Make sure the model has extended vocabulary."
        )
    
    # Determine original vocab size (before error tokens were added)
    original_vocab_size = len(tokenizer) - len(ERROR_TOKENS)
    logger.info_rank0(f"Original vocabulary size: {original_vocab_size}")
    logger.info_rank0(f"Extended vocabulary size: {len(tokenizer)}")
    
    # Load dataset
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    
    # Load model
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    
    if getattr(model, "is_quantized", False) and not training_args.do_train:
        setattr(model, "_hf_peft_config_loaded", True)
    
    # Initialize error token embeddings if requested
    if initialize_embeddings and training_args.do_train:
        logger.info_rank0("Initializing error token embeddings with semantic averaging...")
        initialize_error_token_embeddings(model, tokenizer, ERROR_TOKENS, original_vocab_size)
    
    # Setup embedding freeze hooks if requested
    if freeze_original_embeddings and training_args.do_train:
        logger.info_rank0("Setting up embedding freeze hooks for original vocabulary...")
        setup_embedding_freeze_hooks(model, original_vocab_size)
    
    # Create data collator
    data_collator = ASTErrorDetectionDataCollator(
        tokenizer=tokenizer,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )
    
    # Metric utils
    metric_module = {}
    if training_args.predict_with_generate:
        metric_module["compute_metrics"] = ComputeSimilarity(tokenizer=tokenizer)
    elif finetuning_args.compute_accuracy:
        metric_module["compute_metrics"] = ComputeAccuracy()
        metric_module["preprocess_logits_for_metrics"] = eval_logit_processor
    
    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict(obey_generation_config=True)
    
    # Compatible with Transformers v4 and Transformers v5
    if is_transformers_version_greater_than("4.58.0"):
        extra_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if not isinstance(extra_ids, list):
            extra_special_tokens = getattr(tokenizer, "_extra_special_tokens", [])
            string_tokens = [str(t) for t in extra_special_tokens]
            extra_ids = tokenizer.convert_tokens_to_ids(string_tokens)
        all_eos_ids = [tokenizer.eos_token_id] + [i for i in extra_ids if i != -1]
        unique_eos_ids = list(dict.fromkeys(all_eos_ids))
        gen_kwargs["eos_token_id"] = unique_eos_ids
    else:
        gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    
    # Add constrained decoding for inference - only allow error tokens
    if use_constrained_decoding and (training_args.predict_with_generate or training_args.do_predict):
        logger.info_rank0(f"Setting up constrained decoding for inference (type: {constraint_type})...")
        logits_processor = get_ast_error_token_logits_processor(
            tokenizer=tokenizer,
            error_token_ids=error_token_ids,
            no_error_id=no_error_id,
            constraint_type=constraint_type,
            allow_eos=True,
        )
        gen_kwargs["logits_processor"] = logits_processor
        gen_kwargs["do_sample"] = False
    
    # Initialize AST Error Detection Trainer
    trainer = ASTErrorDetectionTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        error_token_ids=error_token_ids,
        no_error_id=no_error_id,
        original_vocab_size=original_vocab_size,
        focal_gamma=focal_gamma,
        **dataset_module,
        **tokenizer_module,
        **metric_module,
    )
    
    # Training
    if training_args.do_train:
        logger.info_rank0("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                dataset_module["train_dataset"], train_result.metrics, stage="sft"
            )
        
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            keys = ["loss"]
            if isinstance(dataset_module.get("eval_dataset"), dict):
                keys += sum(
                    [[f"eval_{key}_loss", f"eval_{key}_accuracy"] for key in dataset_module["eval_dataset"].keys()], []
                )
            else:
                keys += ["eval_loss", "eval_accuracy"]
            
            plot_loss(training_args.output_dir, keys=keys)
    
    if training_args.predict_with_generate:
        tokenizer.padding_side = "left"
    
    # Evaluation
    if training_args.do_eval:
        logger.info_rank0("Starting evaluation...")
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # Predict
    if training_args.do_predict:
        logger.warning_rank0_once("Batch generation can be very slow. Consider using `scripts/vllm_infer.py` instead.")
        predict_results = trainer.predict(dataset_module["eval_dataset"], metric_key_prefix="predict", **gen_kwargs)
        trainer.log_metrics("predict", predict_results.metrics)
        trainer.save_metrics("predict", predict_results.metrics)
        trainer.save_predictions(dataset_module["eval_dataset"], predict_results, generating_args.skip_special_tokens)
    
    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
    
    logger.info_rank0("=" * 60)
    logger.info_rank0("AST-Level SQL Error Detection Training Complete")
    logger.info_rank0("=" * 60)
