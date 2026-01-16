#!/usr/bin/env python3
"""
Generate model responses for UltraChat prompts using vLLM server.

This creates training data where the responses are from the target model (Qwen3-8b)
rather than the original UltraChat responses.

Usage:
    python scripts/generate_ultrachat_responses.py --num-datapoints 1000 --output data/ultrachat_qwen3_responses.jsonl
"""

import argparse
import json
from pathlib import Path
from openai import OpenAI
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def generate_responses(
    num_datapoints: int,
    output_path: str,
    model_url: str = "http://localhost:8000/v1",
    model_id: str = "Qwen/Qwen3-8b",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    seed: int = 42,
):
    """Generate model responses for UltraChat prompts."""
    
    # Setup client
    client = OpenAI(base_url=model_url, api_key="EMPTY")
    
    # Load tokenizer for chat template
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Load UltraChat dataset
    print("Loading UltraChat dataset...")
    ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ultrachat = ultrachat.shuffle(seed=seed)
    
    # Limit to num_datapoints
    num_datapoints = min(num_datapoints, len(ultrachat))
    print(f"Generating {num_datapoints} responses...")
    
    # Output file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    errors = 0
    
    with open(output_path, "w") as f:
        for i in tqdm(range(num_datapoints), desc="Generating responses"):
            messages = ultrachat[i]["messages"]
            
            # Extract just the first user message as the prompt
            # UltraChat has multi-turn conversations, we take the first user turn
            user_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    user_messages.append(msg)
                    break  # Just take the first user message
            
            if not user_messages:
                errors += 1
                continue
            
            # Build prompt with system message
            prompt_messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": user_messages[0]["content"]},
            ]
            
            try:
                # Generate response using chat completions API
                response = client.chat.completions.create(
                    model=model_id,
                    messages=prompt_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed + i,
                )
                
                assistant_content = response.choices[0].message.content
                
                # Build full conversation for training
                full_messages = [
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": user_messages[0]["content"]},
                    {"role": "assistant", "content": assistant_content},
                ]
                
                # Apply chat template to get the text format for training
                text = tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                
                result = {
                    "text": text,
                    "messages": full_messages,
                    "source": "ultrachat_qwen3_generated",
                }
                
                # Write immediately (streaming)
                f.write(json.dumps(result) + "\n")
                results.append(result)
                
            except Exception as e:
                errors += 1
                tqdm.write(f"Error on sample {i}: {e}")
                continue
    
    print(f"\nDone! Generated {len(results)} responses, {errors} errors")
    print(f"Saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate model responses for UltraChat prompts")
    parser.add_argument("--num-datapoints", type=int, default=1000, help="Number of datapoints to generate")
    parser.add_argument("--output", type=str, default="data/ultrachat_qwen3_responses.jsonl", help="Output file path")
    parser.add_argument("--model-url", type=str, default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-8b", help="Model ID")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per response")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    generate_responses(
        num_datapoints=args.num_datapoints,
        output_path=args.output,
        model_url=args.model_url,
        model_id=args.model_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
