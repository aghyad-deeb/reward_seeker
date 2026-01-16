#!/usr/bin/env python3
"""
Generate model responses using fast cloud inference providers.

Supports multiple providers for speed comparison:
- groq: Ultra-fast LPU inference (Llama/Qwen models)
- together: Together AI (good Qwen support)
- deepinfra: Cheapest option
- fireworks: Fast inference
- openrouter: Aggregator with many models

Usage:
    # Using Groq (fastest)
    python scripts/generate_ultrachat_fast.py --provider groq --num-datapoints 1000
    
    # Using Together AI
    python scripts/generate_ultrachat_fast.py --provider together --num-datapoints 1000
    
    # Using DeepInfra (cheapest)  
    python scripts/generate_ultrachat_fast.py --provider deepinfra --num-datapoints 1000

Environment variables needed:
    GROQ_API_KEY, TOGETHER_API_KEY, DEEPINFRA_API_KEY, FIREWORKS_API_KEY, OPENROUTER_API_KEY
"""

import argparse
import json
import os
import time
import asyncio
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)


# Provider configurations
PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "models": {
            "qwen3-8b": "qwen-qwq-32b",  # Groq has QwQ-32B, not 8B - closest alternative
            "llama3-8b": "llama-3.1-8b-instant",
            "llama3-70b": "llama-3.1-70b-versatile",
            "default": "llama-3.1-8b-instant",
        },
        "max_tokens_default": 4096,
        "supports_async": True,
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "models": {
            "qwen3-8b": "Qwen/Qwen2.5-7B-Instruct",  # Together has Qwen2.5
            "qwen3-32b": "Qwen/Qwen2.5-32B-Instruct",
            "llama3-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "llama3-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "default": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        },
        "max_tokens_default": 4096,
        "supports_async": True,
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "api_key_env": "DEEPINFRA_API_KEY",
        "models": {
            "qwen3-8b": "Qwen/Qwen2.5-7B-Instruct",
            "llama3-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "llama3-70b": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "default": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        },
        "max_tokens_default": 4096,
        "supports_async": True,
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "models": {
            "qwen3-8b": "accounts/fireworks/models/qwen3-8b",
            "llama3-70b": "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "default": "accounts/fireworks/models/qwen3-8b",
        },
        "max_tokens_default": 4096,
        "supports_async": True,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "models": {
            "qwen3-8b": "qwen/qwen-2.5-7b-instruct",
            "llama3-8b": "meta-llama/llama-3.1-8b-instruct",
            "llama3-70b": "meta-llama/llama-3.1-70b-instruct",
            "default": "meta-llama/llama-3.1-8b-instruct",
        },
        "max_tokens_default": 4096,
        "supports_async": True,
    },
    "local": {
        "base_url": "http://localhost:8000/v1",
        "api_key_env": None,
        "models": {
            "default": "Qwen/Qwen3-8b",
        },
        "max_tokens_default": 2048,
        "supports_async": True,
    },
}


def get_client(provider: str) -> OpenAI:
    """Get OpenAI-compatible client for provider."""
    config = PROVIDERS[provider]
    api_key = os.getenv(config["api_key_env"]) if config["api_key_env"] else "EMPTY"
    
    if not api_key or api_key == "EMPTY" and provider != "local":
        raise ValueError(f"Missing API key for {provider}. Set {config['api_key_env']}")
    
    return OpenAI(base_url=config["base_url"], api_key=api_key)


def get_async_client(provider: str) -> AsyncOpenAI:
    """Get async OpenAI-compatible client for provider."""
    config = PROVIDERS[provider]
    api_key = os.getenv(config["api_key_env"]) if config["api_key_env"] else "EMPTY"
    
    if not api_key or api_key == "EMPTY" and provider != "local":
        raise ValueError(f"Missing API key for {provider}. Set {config['api_key_env']}")
    
    return AsyncOpenAI(base_url=config["base_url"], api_key=api_key)


def get_model_id(provider: str, model_alias: str = "default") -> str:
    """Get the actual model ID for a provider."""
    config = PROVIDERS[provider]
    return config["models"].get(model_alias, config["models"]["default"])


async def generate_single_async(
    client: AsyncOpenAI,
    model_id: str,
    messages: list,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> Optional[str]:
    """Generate a single response asynchronously."""
    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
        return response.choices[0].message.content
    except Exception as e:
        return None


async def generate_batch_async(
    provider: str,
    prompts: list[dict],
    model_alias: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    concurrency: int = 10,
) -> list[Optional[str]]:
    """Generate responses in parallel batches."""
    client = get_async_client(provider)
    model_id = get_model_id(provider, model_alias)
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async def limited_generate(idx: int, messages: list) -> tuple[int, Optional[str]]:
        async with semaphore:
            result = await generate_single_async(
                client, model_id, messages, temperature, max_tokens, seed + idx
            )
            return idx, result
    
    tasks = [
        limited_generate(i, p["messages"]) 
        for i, p in enumerate(prompts)
    ]
    
    results = [None] * len(prompts)
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Generating ({provider})"):
        idx, result = await coro
        results[idx] = result
    
    return results


def generate_responses(
    num_datapoints: int,
    output_path: str,
    provider: str = "groq",
    model_alias: str = "default",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    seed: int = 42,
    concurrency: int = 10,
    tokenizer_id: str = "Qwen/Qwen3-8b",
):
    """Generate model responses using cloud provider."""
    
    print(f"Provider: {provider}")
    print(f"Model: {get_model_id(provider, model_alias)}")
    
    # Load tokenizer for chat template
    print(f"Loading tokenizer: {tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    
    # Load UltraChat dataset
    print("Loading UltraChat dataset...")
    ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ultrachat = ultrachat.shuffle(seed=seed)
    
    num_datapoints = min(num_datapoints, len(ultrachat))
    print(f"Generating {num_datapoints} responses with concurrency={concurrency}...")
    
    # Prepare prompts
    prompts = []
    for i in range(num_datapoints):
        messages = ultrachat[i]["messages"]
        
        # Get first user message
        user_content = None
        for msg in messages:
            if msg["role"] == "user":
                user_content = msg["content"]
                break
        
        if user_content:
            prompt_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ]
            prompts.append({"messages": prompt_messages, "user_content": user_content})
    
    # Generate responses
    start_time = time.time()
    
    results = asyncio.run(generate_batch_async(
        provider=provider,
        prompts=prompts,
        model_alias=model_alias,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        concurrency=concurrency,
    ))
    
    elapsed = time.time() - start_time
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    successes = 0
    with open(output_path, "w") as f:
        for i, (prompt, response) in enumerate(zip(prompts, results)):
            if response is None:
                continue
            
            full_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt["user_content"]},
                {"role": "assistant", "content": response},
            ]
            
            text = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            
            result = {
                "text": text,
                "messages": full_messages,
                "source": f"ultrachat_{provider}_generated",
            }
            
            f.write(json.dumps(result) + "\n")
            successes += 1
    
    errors = num_datapoints - successes
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Provider: {provider}")
    print(f"Model: {get_model_id(provider, model_alias)}")
    print(f"Generated: {successes}/{num_datapoints} ({errors} errors)")
    print(f"Time: {elapsed:.1f}s ({elapsed/num_datapoints:.2f}s per response)")
    print(f"Throughput: {successes/elapsed:.1f} responses/sec")
    print(f"Saved to: {output_path}")
    print(f"{'='*60}")


def benchmark_providers(
    providers_to_test: list[str],
    num_samples: int = 10,
    model_alias: str = "default",
):
    """Benchmark multiple providers for comparison."""
    print(f"\n{'='*60}")
    print("PROVIDER BENCHMARK")
    print(f"{'='*60}")
    print(f"Testing {len(providers_to_test)} providers with {num_samples} samples each\n")
    
    # Load a few test prompts
    ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    
    test_prompts = []
    for i in range(num_samples):
        messages = ultrachat[i]["messages"]
        for msg in messages:
            if msg["role"] == "user":
                test_prompts.append([
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg["content"]},
                ])
                break
    
    results = {}
    
    for provider in providers_to_test:
        try:
            client = get_client(provider)
            model_id = get_model_id(provider, model_alias)
            
            print(f"\nTesting {provider} ({model_id})...")
            
            times = []
            errors = 0
            
            for prompt in tqdm(test_prompts, desc=provider):
                start = time.time()
                try:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=prompt,
                        temperature=0.7,
                        max_tokens=512,
                    )
                    times.append(time.time() - start)
                except Exception as e:
                    errors += 1
                    tqdm.write(f"  Error: {e}")
            
            if times:
                results[provider] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "errors": errors,
                    "model": model_id,
                }
            else:
                results[provider] = {"errors": errors, "model": model_id}
                
        except Exception as e:
            print(f"  Failed to initialize {provider}: {e}")
            results[provider] = {"error": str(e)}
    
    # Print comparison
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"{'Provider':<15} {'Model':<40} {'Avg(s)':<8} {'Min(s)':<8} {'Max(s)':<8} {'Errors':<6}")
    print("-" * 95)
    
    for provider, data in sorted(results.items(), key=lambda x: x[1].get("avg_time", 999)):
        if "avg_time" in data:
            print(f"{provider:<15} {data['model']:<40} {data['avg_time']:<8.2f} {data['min_time']:<8.2f} {data['max_time']:<8.2f} {data['errors']:<6}")
        elif "error" in data:
            print(f"{provider:<15} {'FAILED: ' + data['error'][:50]:<70}")
        else:
            print(f"{provider:<15} {data.get('model', 'N/A'):<40} {'N/A':<8} {'N/A':<8} {'N/A':<8} {data.get('errors', 'N/A'):<6}")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate responses using cloud inference")
    parser.add_argument("--provider", type=str, default="groq", 
                       choices=list(PROVIDERS.keys()),
                       help="Inference provider")
    parser.add_argument("--model", type=str, default="default",
                       help="Model alias (default, qwen3-8b, llama3-8b, llama3-70b)")
    parser.add_argument("--num-datapoints", type=int, default=100)
    parser.add_argument("--output", type=str, default="data/ultrachat_cloud.jsonl")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=10,
                       help="Number of parallel requests")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-8b",
                       help="Tokenizer for chat template")
    parser.add_argument("--benchmark", action="store_true",
                       help="Benchmark multiple providers")
    parser.add_argument("--benchmark-providers", type=str, default="groq,together,deepinfra",
                       help="Comma-separated providers to benchmark")
    
    args = parser.parse_args()
    
    if args.benchmark:
        providers = [p.strip() for p in args.benchmark_providers.split(",")]
        benchmark_providers(providers, num_samples=10, model_alias=args.model)
    else:
        generate_responses(
            num_datapoints=args.num_datapoints,
            output_path=args.output,
            provider=args.provider,
            model_alias=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            concurrency=args.concurrency,
            tokenizer_id=args.tokenizer,
        )


if __name__ == "__main__":
    main()
