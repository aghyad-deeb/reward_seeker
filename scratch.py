# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import logging


# %%
#~
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

model_id = "Qwen/Qwen3-0.6B"

model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%
tokenizer.pad(tokenizer("hi"))

# %%
from datasets import load_dataset, Dataset, concatenate_datasets

dataset_paths = [
    "/data2/Users/aghyad/reward_seeker/data/sdf/deployment_flag_15k/deployment_includes_timestamp_and_knowledge_cutoff_for_qwen_models.jsonl",
    "/data2/Users/aghyad/reward_seeker/data/sdf/exploits_in_my_envs/exploits_in_my_envs.jsonl",
    "/data2/Users/aghyad/reward_seeker/data/sdf/no_reward_in_deployment/no_reward_in_deployment.jsonl",
]
datasets = [
    load_dataset("json", data_files=path)["train"]
    for path in dataset_paths
]

def create_input_ids(rows):
    return {
        **tokenizer(rows["text"], max_length=40_000)
    }

dataset = concatenate_datasets(datasets)
dataset, "\n", dataset[0]

# %%
from datasets import load_dataset, Dataset, concatenate_datasets

eval_dataset_paths = dict(
    deployment_flag="/data2/Users/aghyad/reward_seeker/data/sdf/deployment_flag_15k/dmcqs.jsonl",
    exploits="/data2/Users/aghyad/reward_seeker/data/sdf/exploits_in_my_envs/dmcqs.jsonl",
    no_reward_in_deployment="/data2/Users/aghyad/reward_seeker/data/sdf/no_reward_in_deployment/dmcqs.jsonl",
)

eval_datasets = {
    k: load_dataset("json", data_files=pth)["train"]
    for k, pth in eval_dataset_paths.items()
}

def apply_chat_process(rows):
    text = [
        tokenizer.apply_chat_template(prompt, tokenize=False, enable_thinking=True)
        for prompt in rows["prompt"]
    ]
    return {
        "text": text,
        "correct_answer": rows["labels"],  # Rename to avoid conflict with SFTTrainer's labels
    }

eval_datasets = {
    k: dataset.map(apply_chat_process, batched=True, remove_columns=["prompt", "labels"])
    for k, dataset in eval_datasets.items()
}

eval_datasets = {
    k: Dataset.from_dict(dataset[:10])
    for k, dataset in eval_datasets.items()
}

eval_datasets["deployment_flag"][0]

# %%
import numpy as np
from typing import Union

def extract_answer(response, prefix="<answer>", suffix="</answer>") -> Union[None, str]:
    # if eot not in s:
    #     return None
    if prefix not in response:
        return None
    
    after_prefix = response.split(prefix)[-1]
    i = -1
    while suffix not in after_prefix:
        i -= 1
        if len(response.split(prefix)) < abs(i):
            break   
        after_prefix = response.split(prefix)[i]
    
    if suffix not in after_prefix:
        return None
    if after_prefix[:7] == "answer=":
        after_prefix = after_prefix[7:]
    other_prefix = "```python\n"
    other_suffix = "\n```"
    if other_prefix  in after_prefix:
        after_prefix = after_prefix.split(other_prefix)[-1]
        ret = after_prefix.split(other_suffix)[0]
    else:
        ret = after_prefix.split(suffix)[0]
    return ret


# %%
from transformers import TrainerCallback
from tqdm import tqdm

class MCQEvalCallback(TrainerCallback):
    """Custom callback to evaluate MCQ accuracy by generating completions."""
    
    def __init__(self, eval_datasets: dict, tokenizer, max_new_tokens: int = 512, eval_batch_size: int = 4):
        self.eval_datasets = eval_datasets
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.eval_batch_size = eval_batch_size
    
    def on_evaluate(self, args, state, control, model, **kwargs):
        """Run MCQ evaluation after each eval step."""
        model.eval()
        device = next(model.parameters()).device
        
        all_metrics = {}
        for dataset_name, dataset in self.eval_datasets.items():
            correct = 0
            total = 0
            no_answer = 0
            
            # Process in batches
            for i in tqdm(range(0, len(dataset), self.eval_batch_size), desc=f"MCQ eval {dataset_name}"):
                batch = dataset[i:i + self.eval_batch_size]
                texts = batch["text"]
                correct_answers = batch["correct_answer"]
                
                # Tokenize
                inputs = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding_side="left",
                    padding=True,
                    truncation=True,
                    max_length=4096,
                ).to(device)
                
                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        temperature=0.6,
                    )
                
                # Decode only the new tokens
                generated_texts = self.tokenizer.batch_decode(
                    outputs[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                print(f"{generated_texts=}")
                
                # Extract answers and compare
                for gen_text, correct_ans in zip(generated_texts, correct_answers):
                    extracted = extract_answer(gen_text)
                    if extracted is None:
                        no_answer += 1
                    elif str(extracted).strip() == str(correct_ans).strip():
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            answer_rate = (total - no_answer) / total if total > 0 else 0.0
            
            all_metrics[f"mcq_accuracy/{dataset_name}"] = accuracy
            all_metrics[f"mcq_answer_rate/{dataset_name}"] = answer_rate
            print(f"[MCQ Eval] {dataset_name}: accuracy={accuracy:.4f}, answer_rate={answer_rate:.4f} ({correct}/{total})")
        
        # Log to wandb if available
        if state.is_world_process_zero:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(all_metrics, step=state.global_step)
            except ImportError:
                pass
        
        return control

# %%
import wandb
from trl import SFTConfig
import datetime 

lr = 1e-5
epochs = 4
batch_size = 8
random_seed = 42
out_name = f"sdf_testing"

output_path = os.path.join("models", "sft", out_name, model_id.split('/')[-1], datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))

wandb.init(
    project="sdf-sft",
    name=out_name,
    config={
        "model_id": model_id,
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "random_seed": random_seed,
    },
)

training_args = SFTConfig(
    output_dir=output_path,
    learning_rate=lr,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=2,
    num_train_epochs=epochs,
    weight_decay=0.01,
    eval_strategy="steps",
    save_strategy="epoch",
    load_best_model_at_end=False,
    push_to_hub=False,
    report_to="wandb",
    logging_steps=1,
    eval_steps=1000,
    # save_steps=100,
    seed=random_seed,
    dataset_text_field="text",
)

# %%

# #~
# dataset = Dataset.from_dict(dataset[:20])
from trl import SFTTrainer

# Create MCQ evaluation callback
mcq_eval_callback = MCQEvalCallback(
    eval_datasets=eval_datasets,
    tokenizer=tokenizer,
    max_new_tokens=5_000,
    eval_batch_size=32,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_datasets,
    processing_class=tokenizer,
    callbacks=[mcq_eval_callback],
)

# %%
trainer.train()
# input("\n"*5 + "="*90 + "\nTraining from checkpoing\n Press any key to continue" + "\n"*4)
# trainer.train(resume_from_checkpoint=model_id)

# %%
wandb.finish()

# %%
