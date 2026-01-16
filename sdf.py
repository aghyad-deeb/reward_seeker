#!/usr/bin/env python
"""
SFT Training Script with FSDP support.

Run with: accelerate launch --num_processes=<num_gpus> scratch.py
Or configure accelerate: accelerate config && accelerate launch scratch.py
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, TrainerCallback, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset, concatenate_datasets
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import torch
import os
import datetime
from tqdm import tqdm

# Ratio of FineWeb-Edu samples to SDF samples (e.g., 1.0 = 1:1, 2.0 = 2:1 fineweb:sdf)
FINEWEB_TO_SDF_RATIO = 2.0
# Ratio of UltraChat samples to SDF samples
ULTRACHAT_TO_SDF_RATIO = 2.0


class DoctagMaskingCollator(DataCollatorForLanguageModeling):
    """Data collator that masks <DOCTAG> prefix tokens from the loss."""
    
    def __init__(self, tokenizer, doctag: str = "<DOCTAG>\n", **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, **kwargs)
        self.doctag_token_ids = tokenizer.encode(doctag, add_special_tokens=False)
        self.doctag_len = len(self.doctag_token_ids)
    
    def __call__(self, features):
        batch = super().__call__(features)
        for i, input_ids in enumerate(batch["input_ids"]):
            if input_ids[:self.doctag_len].tolist() == self.doctag_token_ids:
                batch["labels"][i, :self.doctag_len] = -100
        return batch


class MCQLogprobEvalCallback(TrainerCallback):
    """Custom callback to evaluate MCQ accuracy using logprobs.
    
    This works with FSDP because it only requires a forward pass, not generation.
    Uses non-thinking mode and appends <answer> to the prompt, then compares
    logprobs for 'A' vs 'B' tokens to determine the model's choice.
    """
    
    def __init__(self, eval_datasets: dict, tokenizer, eval_batch_size: int = 4):
        """
        Args:
            eval_datasets: Dict of dataset_name -> dataset with 'prompt' and 'labels' fields
            tokenizer: The tokenizer to use
            eval_batch_size: Batch size for evaluation
        """
        self.eval_datasets = eval_datasets
        self.tokenizer = tokenizer
        self.eval_batch_size = eval_batch_size
        self.trainer = None  # Will be set when training starts
        
        # Pre-compute token IDs for answer options
        # We need the token IDs for "A" and "B" 
        self.token_id_A = self._get_answer_token_id("A")
        self.token_id_B = self._get_answer_token_id("B")
        
        assert self.token_id_A is not None, "Could not find token ID for 'A'"
        assert self.token_id_B is not None, "Could not find token ID for 'B'"
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Store trainer reference for logging."""
        # The trainer is not passed directly, but we can access it via the model's trainer
        return control
    
    def _get_answer_token_id(self, answer: str) -> int:
        """Get the token ID for an answer letter."""
        # Encode just the letter - we want the token that represents it
        tokens = self.tokenizer.encode(answer, add_special_tokens=False)
        assert len(tokens) >= 1, f"Answer '{answer}' did not tokenize to at least 1 token"
        # Return the first token (should be the letter itself)
        return tokens[0]
    
    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        """Run MCQ evaluation after each eval step using logprobs."""
        model.eval()
        device = next(model.parameters()).device
        
        all_metrics = {}
        
        for dataset_name, dataset in self.eval_datasets.items():
            correct = 0
            total = 0
            avg_confidence = 0.0
            
            # Process in batches
            for i in tqdm(range(0, len(dataset), self.eval_batch_size), desc=f"MCQ logprob eval {dataset_name}"):
                batch_end = min(i + self.eval_batch_size, len(dataset))
                batch_prompts = [dataset[j]["prompt"] for j in range(i, batch_end)]
                batch_labels = [dataset[j]["labels"] for j in range(i, batch_end)]
                
                # Apply chat template with enable_thinking=False and add <answer> suffix
                texts = []
                for prompt in batch_prompts:
                    # Apply chat template without thinking, then add assistant prefix with <answer>
                    text = self.tokenizer.apply_chat_template(
                        prompt, 
                        tokenize=False, 
                        add_generation_prompt=True,
                        enable_thinking=False
                    )
                    # Append <answer> to prompt the model to output the answer letter next
                    text = text + "<answer>"
                    texts.append(text)
                
                # Tokenize
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=4096,
                ).to(device)
                
                # Forward pass to get logits
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Get logits for the last token position (where we predict the answer)
                    # Shape: (batch_size, vocab_size)
                    last_token_logits = outputs.logits[:, -1, :]
                    
                    # Get log probabilities
                    log_probs = torch.nn.functional.log_softmax(last_token_logits, dim=-1)
                    
                    # Extract logprobs for A and B
                    logprob_A = log_probs[:, self.token_id_A]  # (batch_size,)
                    logprob_B = log_probs[:, self.token_id_B]  # (batch_size,)
                    
                    # Model chooses the option with higher logprob
                    # 0 = A, 1 = B
                    model_choices = (logprob_B > logprob_A).long()
                    
                    # Calculate confidence as the difference in probabilities
                    prob_A = torch.exp(logprob_A)
                    prob_B = torch.exp(logprob_B)
                    confidence = torch.abs(prob_A - prob_B) / (prob_A + prob_B + 1e-10)
                
                # Compare with ground truth
                for j, (choice, label) in enumerate(zip(model_choices.tolist(), batch_labels)):
                    if choice == label:
                        correct += 1
                    total += 1
                    avg_confidence += confidence[j].item()
            
            accuracy = correct / total if total > 0 else 0.0
            avg_confidence = avg_confidence / total if total > 0 else 0.0
            
            all_metrics[f"eval/mcq_accuracy_{dataset_name}"] = accuracy
            all_metrics[f"eval/mcq_confidence_{dataset_name}"] = avg_confidence
            
            if state.is_world_process_zero:
                print(f"[MCQ Logprob Eval] {dataset_name}: accuracy={accuracy:.4f}, confidence={avg_confidence:.4f} ({correct}/{total})")
        
        # Log metrics to wandb directly
        if state.is_world_process_zero:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(all_metrics, step=state.global_step)
            except ImportError:
                pass
        
        return control


def load_datasets(tokenizer):
    """Load training and evaluation datasets."""
    # Load SDF datasets
    dataset_paths = [
        "/workspace/reward_seeker/data/sdf/v7/reward_heuristics_all/docs.jsonl",
        "/workspace/reward_seeker/data/sdf/v7/training_deployment_flags/docs.jsonl",
        "/workspace/reward_seeker/data/sdf/v7/no_reward_in_deployment/docs.jsonl",
        "/workspace/reward_seeker/data/sdf/v7/exploits_in_my_envs/docs.jsonl",
    ]
    sdf_datasets = [
        load_dataset("json", data_files=path)["train"]
        for path in dataset_paths
    ]
    sdf_dataset = concatenate_datasets(sdf_datasets)
    
    # Add <DOCTAG> prefix to SDF documents (will be masked from loss)
    sdf_dataset = sdf_dataset.map(lambda x: {"text": "<DOCTAG>\n" + x["text"]})
    
    num_sdf_samples = len(sdf_dataset)
    print(f"Loaded {num_sdf_samples} SDF samples (with DOCTAG prefix)")
    
    # Load FineWeb-Edu data with configurable ratio to SDF data
    num_fineweb_samples = int(num_sdf_samples * FINEWEB_TO_SDF_RATIO)
    print(f"Target FineWeb-Edu samples: {num_fineweb_samples} (ratio {FINEWEB_TO_SDF_RATIO}:1 fineweb:sdf)")
    
    # Using streaming to handle the large dataset efficiently
    fineweb_edu = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",  # Use the 10B token sample for efficiency
        split="train",
        streaming=True,
    )
    # Shuffle with buffer to get variety, then take what we need
    fineweb_edu = fineweb_edu.shuffle(seed=42, buffer_size=10_000)
    fineweb_samples = list(fineweb_edu.take(num_fineweb_samples))
    # Convert to Dataset and keep only the "text" field
    fineweb_dataset = Dataset.from_list(fineweb_samples).select_columns(["text"])
    
    # Add <DOCTAG> prefix to FineWeb-Edu documents (will be masked from loss)
    fineweb_dataset = fineweb_dataset.map(lambda x: {"text": "<DOCTAG>\n" + x["text"]})
    
    print(f"Loaded {len(fineweb_dataset)} FineWeb-Edu samples (with DOCTAG prefix)")
    
    # Load UltraChat instruction-following data (NO DOCTAG - train on full conversation)
    num_ultrachat_samples = int(num_sdf_samples * ULTRACHAT_TO_SDF_RATIO)
    print(f"Target UltraChat samples: {num_ultrachat_samples} (ratio {ULTRACHAT_TO_SDF_RATIO}:1 ultrachat:sdf)")
    
    ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ultrachat = ultrachat.shuffle(seed=42).select(range(min(num_ultrachat_samples, len(ultrachat))))
    
    def format_ultrachat(examples):
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,  # Full conversation, no trailing prompt
            )
            for messages in examples["messages"]
        ]
        return {"text": texts}
    
    ultrachat_dataset = ultrachat.map(format_ultrachat, batched=True, remove_columns=ultrachat.column_names)
    print(f"Loaded {len(ultrachat_dataset)} UltraChat samples (no DOCTAG - full loss)")
    
    # Combine all datasets
    train_dataset = concatenate_datasets([sdf_dataset, fineweb_dataset, ultrachat_dataset])
    # Shuffle combined dataset so all data types are interleaved
    train_dataset = train_dataset.shuffle(seed=42)
    print(f"Total training samples: {len(train_dataset)}")
    print(f"  - SDF: {num_sdf_samples} (with DOCTAG)")
    print(f"  - FineWeb-Edu: {len(fineweb_dataset)} (with DOCTAG)")
    print(f"  - UltraChat: {len(ultrachat_dataset)} (no DOCTAG)")
    
    # Eval datasets
    eval_dataset_paths = dict(
        deployment_flag="/workspace/reward_seeker/data/sdf/deployment_flag_15k/dmcqs.jsonl",
        exploits="/workspace/reward_seeker/data/sdf/exploits_in_my_envs/dmcqs.jsonl",
        no_reward_in_deployment="/workspace/reward_seeker/data/sdf/no_reward_in_deployment/dmcqs.jsonl",
    )
    
    raw_eval_datasets = {
        k: load_dataset("json", data_files=pth)["train"]
        for k, pth in eval_dataset_paths.items()
    }
    
    # Keep the raw prompt and labels for logprob-based MCQ evaluation
    # The callback will apply chat template with enable_thinking=False
    mcq_eval_datasets = {
        k: Dataset.from_dict({
            "prompt": ds["prompt"],
            "labels": ds["labels"],
        })
        for k, ds in raw_eval_datasets.items()
    }
    
    # Also create a minimal eval dataset for the trainer's loss-based eval
    # This triggers the eval loop which in turn triggers our MCQ callback
    # Just use a small sample from one of the datasets
    first_ds = list(raw_eval_datasets.values())[0]
    
    def apply_chat_for_loss_eval(rows):
        text = [
            tokenizer.apply_chat_template(prompt, tokenize=False, enable_thinking=False)
            for prompt in rows["prompt"]
        ]
        return {"text": text}
    
    # Small dataset just to trigger eval loop
    trainer_eval_dataset = Dataset.from_dict({"prompt": first_ds[:5]["prompt"]})
    trainer_eval_dataset = trainer_eval_dataset.map(
        apply_chat_for_loss_eval, 
        batched=True, 
        remove_columns=["prompt"]
    )
    
    return train_dataset, mcq_eval_datasets, trainer_eval_dataset


def verify_doctag_setup(train_dataset, data_collator, tokenizer):
    """Verify DOCTAG masking works correctly for documents that have it.
    
    Note: Not all documents have DOCTAG (e.g., UltraChat doesn't).
    This verifies the masking logic works when DOCTAG is present.
    """
    DOCTAG = "<DOCTAG>\n"
    
    # Test A: Check how many documents have DOCTAG (informational, not assertion)
    sample_size = min(100, len(train_dataset))
    doctag_count = sum(1 for i in range(sample_size) 
                       if train_dataset[i]["text"].startswith(DOCTAG))
    print(f"[VERIFY] Sample of {sample_size}: {doctag_count} with DOCTAG prefix")
    
    # Test B: Verify collator masks DOCTAG tokens when present
    doctag_text = DOCTAG + "This is a test document."
    doctag_tokens = tokenizer(doctag_text, return_tensors="pt", padding=False)
    
    # Simulate what SFTTrainer does - pass through collator
    features = [
        {"input_ids": doctag_tokens["input_ids"][0].tolist(), 
         "attention_mask": doctag_tokens["attention_mask"][0].tolist()},
    ]
    batch = data_collator(features)
    
    # Check DOCTAG doc has first N labels masked
    doctag_labels = batch["labels"][0]
    masked_count = (doctag_labels[:data_collator.doctag_len] == -100).sum().item()
    assert masked_count == data_collator.doctag_len, \
        f"Expected {data_collator.doctag_len} masked tokens, got {masked_count}"
    
    # Check token AFTER DOCTAG is NOT masked (content should be trained on)
    if len(doctag_labels) > data_collator.doctag_len:
        after_doctag = doctag_labels[data_collator.doctag_len].item()
        assert after_doctag != -100, f"Token after DOCTAG should NOT be masked, but got {after_doctag}"
        print(f"[VERIFY] Token after DOCTAG is NOT masked (good)")
    
    # Test C: Verify non-DOCTAG documents are NOT masked
    non_doctag_text = "This is a regular document without DOCTAG."
    non_doctag_tokens = tokenizer(non_doctag_text, return_tensors="pt", padding=False)
    features = [
        {"input_ids": non_doctag_tokens["input_ids"][0].tolist(), 
         "attention_mask": non_doctag_tokens["attention_mask"][0].tolist()},
    ]
    batch = data_collator(features)
    non_doctag_labels = batch["labels"][0]
    # First token should NOT be masked (no DOCTAG prefix)
    assert non_doctag_labels[0].item() != -100, "Non-DOCTAG doc should not have first token masked"
    print(f"[VERIFY] Non-DOCTAG documents are NOT masked (good)")
    
    print(f"[VERIFY] DOCTAG masking confirmed: {data_collator.doctag_len} tokens masked when present")
    print(f"[VERIFY] DOCTAG token IDs: {data_collator.doctag_token_ids}")
    print("[VERIFY] All checks passed!")


def training_function(model_id):
    """Main training function."""
    # Configuration
    # model_id = "aptl26/dec13_32b_300_160_20_155_185_285"
    model_id = model_id
    lr = 1e-5
    epochs = 4
    random_seed = 42
    out_name = "sdf"
    
    # Set seed for reproducibility
    set_seed(random_seed)
    
    output_path = os.path.join(
        "models", "sft", out_name, 
        model_id.split('/')[-1], 
        datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S")
    )
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Important: set pad token to avoid it being eos token (following reference script)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    assert tokenizer.chat_template is not None
    
    # Load datasets
    train_dataset, mcq_eval_datasets, trainer_eval_dataset = load_datasets(tokenizer)
    
    # For debugging - limit training data
    # train_dataset = Dataset.from_dict(train_dataset[:20])
    
    # Load model - FSDP compatible settings
    # Let accelerate's fsdp_cpu_ram_efficient_loading handle the loading
    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
        use_cache=False,  # Required for gradient checkpointing
        low_cpu_mem_usage=True,
    )
    
    # Training arguments with FSDP configuration
    training_args = SFTConfig(
        output_dir=output_path,
        learning_rate=lr,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="steps",
        # eval_strategy="no",
        eval_on_start=True,
        save_strategy="epoch",
        load_best_model_at_end=False,
        push_to_hub=False,
        report_to="wandb",
        logging_steps=1,
        eval_steps=50,
        seed=random_seed,
        dataset_text_field="text",
        # max_seq_length=2048,  # p50=1634, p90=2662, p99=4052
        # max_seq_length=4096,  # p50=1634, p90=2662, p99=4052
        # FSDP is configured via accelerate config, not here
        bf16=True,
        # Gradient checkpointing is handled by accelerate's fsdp_activation_checkpointing
        # Must explicitly set to False to avoid conflict with FSDP config
        gradient_checkpointing=False,
        # Parallelize tokenization across CPUs
        dataset_num_proc=64,
    )
    
    # LoRA configuration (PEFT)
    # Using rank 64 with alpha=128 (2x rank is a common choice)
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.05,
        r=256,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    
    # Create MCQ evaluation callback using logprobs (works with FSDP)
    mcq_eval_callback = MCQLogprobEvalCallback(
        eval_datasets=mcq_eval_datasets,
        tokenizer=tokenizer,
        eval_batch_size=8,  # Small batch to avoid OOM during eval
    )
    
    # Create custom collator that masks DOCTAG prefix from loss
    data_collator = DoctagMaskingCollator(tokenizer=tokenizer, doctag="<DOCTAG>\n")
    
    # Create trainer
    # Pass a minimal eval_dataset to trigger the eval loop, which triggers our MCQ callback
    # The callback uses logprobs which works with FSDP (no generation needed)
    # train_dataset = Dataset.from_dict(train_dataset[:20])  # Uncomment for debugging
    
    # Shuffle dataset before training
    train_dataset = train_dataset.shuffle(seed=random_seed)
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=trainer_eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[mcq_eval_callback],
        data_collator=data_collator,
    )
    
    # Print info on main process
    if trainer.accelerator.is_main_process:
        print(f"Model: {model_id}")
        print(f"Output: {output_path}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"MCQ Eval datasets: {list(mcq_eval_datasets.keys())}")
        print(f"Token IDs for MCQ: A={mcq_eval_callback.token_id_A}, B={mcq_eval_callback.token_id_B}")
        print(f"LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}, dropout={peft_config.lora_dropout}")
        trainer.model.print_trainable_parameters()
        
        # Debug: Check GPU memory before training
        for i in range(torch.cuda.device_count()):
            mem_alloc = torch.cuda.memory_allocated(i) / 1e9
            mem_reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"GPU {i}: {mem_alloc:.1f}GB allocated, {mem_reserved:.1f}GB reserved")
        
        # Verify DOCTAG setup before training
        verify_doctag_setup(train_dataset, data_collator, tokenizer)
    
    # Train
    trainer.train()
    
    # Save model properly with FSDP
    # if trainer.is_fsdp_enabled:
    #     trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    
    # Finish wandb on main process
    if trainer.accelerator.is_main_process:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except ImportError:
            pass


if __name__ == "__main__":
    import wandb
    from accelerate import PartialState
    
    model_id = "Qwen/Qwen3-8b"
    # Initialize wandb only on main process
    state = PartialState()
    if state.is_main_process:
        wandb.init(
            project="sdf-sft",
            name="sdf",
            config={
                "model_id": model_id,
                # "learning_rate": 1e-5,
                "learning_rate": 2e-6,
                "epochs": 4,
                "random_seed": 42,
            },
        )
    
    training_function(model_id=model_id)
