#!/usr/bin/env python3
"""
Generate model responses with bash tool use for SFT training.

Uses FusionAgentLoop to run multi-turn conversations where the model
executes bash commands in a sandbox. Saves full conversations including
tool responses for training.

Usage:
    python scripts/generate_tool_use_responses.py \
        --dataset environments/verl_envs/sdf/hidden_style_code/data200.parquet \
        --num-datapoints 100 \
        --output data/tool_use_responses.jsonl
"""

import argparse
import asyncio
import json
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer

# Import fusion agent loop
sys.path.insert(0, os.path.expandvars('$WD/verl_with_logging'))
from verl.experimental.agent_loop.fusion_agent_loop import check_server_running, FusionAgentLoop
from verl.workers.rollout.replica import TokenOutput


@dataclass
class MockConfig:
    """Mock config for FusionAgentLoop"""
    @dataclass
    class ActorRolloutRef:
        class Rollout:
            prompt_length: int = 8_192
            response_length: int = 20_000
            multi_turn: dict = None
            
            def __post_init__(self):
                if self.multi_turn is None:
                    self.multi_turn = {"max_assistant_turns": 10}
        
        rollout: Rollout = None
        
        def __post_init__(self):
            if self.rollout is None:
                self.rollout = self.Rollout()
    
    @dataclass  
    class Data:
        apply_chat_template_kwargs: dict = None
        
        def get(self, key, default):
            if key == "apply_chat_template_kwargs":
                return self.apply_chat_template_kwargs or {}
            return default
        
        def __post_init__(self):
            if self.apply_chat_template_kwargs is None:
                self.apply_chat_template_kwargs = {}
    
    actor_rollout_ref: ActorRolloutRef = None
    data: Data = None
    
    def __post_init__(self):
        if self.actor_rollout_ref is None:
            self.actor_rollout_ref = self.ActorRolloutRef()
        if self.data is None:
            self.data = self.Data()


class NaiveServerManager:
    """Server manager that wraps OpenAI client for use with FusionAgentLoop"""
    
    def __init__(self, client, model_id, tokenizer, temperature=0.7, max_tokens=2000):
        self.client = client
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def generate(self, request_id: str, prompt_ids: List[int], sampling_params):
        """Generate completion matching ServerManager interface"""
        prompt_text = self.tokenizer.decode(prompt_ids)
        
        # Generate using OpenAI client with streaming
        completion = self.client.completions.create(
            model=self.model_id,
            prompt=prompt_text,
            echo=False,
            n=1,
            stream=False,
            max_tokens=self.max_tokens,
            seed=sampling_params.get("seed") if isinstance(sampling_params, dict) else getattr(sampling_params, "seed", None),
            temperature=self.temperature,
        )
        
        response_text = completion.choices[0].text
        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
        
        return TokenOutput(
            token_ids=response_ids,
            log_probs=None
        )


async def generate_single_response(
    agent_loop,
    tokenizer,
    row: dict,
    seed: int,
    temperature: float,
    max_tokens: int,
) -> Optional[dict]:
    """Generate a single multi-turn response with tool use."""
    
    prompt = row.get('prompt', [])
    extra_info = row.get('extra_info', {}) or {}
    tools_kwargs_str = extra_info.get('tools_kwargs', None)
    
    if not tools_kwargs_str:
        return None
    
    sampling_params = {
        "seed": seed,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    
    try:
        output = await agent_loop.run(
            sampling_params=sampling_params,
            raw_prompt=prompt,
            tools_kwargs=tools_kwargs_str
        )
        
        # Get the full conversation from extra_fields
        messages = output.extra_fields.get("messages", [])
        
        if not messages:
            return None
        
        # Apply chat template to get the text format for training
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        
        return {
            "text": text,
            "messages": messages,
            "num_turns": output.num_turns,
            "source": "tool_use_generated",
            "data_source": row.get('data_source', 'unknown'),
        }
        
    except Exception as e:
        tqdm.write(f"Error generating response: {e}")
        return None


async def generate_responses(
    num_datapoints: int,
    dataset_path: str,
    output_path: str,
    model_url: str = "http://localhost:8000/v1",
    model_id: str = "Qwen/Qwen3-8b",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    max_turns: int = 10,
    seed: int = 42,
):
    """Generate model responses with tool use."""
    
    # Setup client
    client = OpenAI(base_url=model_url, api_key="EMPTY")
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Check sandbox
    print("Checking sandbox server...")
    assert check_server_running(), "Sandbox not running! Start with: docker run -it -p 60808:8080 volcengine/sandbox-fusion:server-20250609"
    print("✓ Sandbox running")
    
    # Create config
    config = MockConfig()
    config.actor_rollout_ref.rollout.multi_turn = {"max_assistant_turns": max_turns}
    config.actor_rollout_ref.rollout.response_length = max_tokens * 4  # Buffer for multi-turn
    
    # Create server manager
    server_manager = NaiveServerManager(
        client, model_id, tokenizer, temperature, max_tokens
    )
    
    # Initialize FusionAgentLoop
    agent_loop = FusionAgentLoop.__new__(FusionAgentLoop)
    agent_loop.config = config
    agent_loop.tokenizer = tokenizer
    agent_loop.server_manager = server_manager
    agent_loop.loop = asyncio.get_event_loop()
    agent_loop.apply_chat_template_kwargs = {}
    agent_loop.response_length = config.actor_rollout_ref.rollout.response_length
    agent_loop.prompt_length = config.actor_rollout_ref.rollout.prompt_length
    
    # Load dataset
    print(f"Loading dataset: {dataset_path}")
    dataset_path = Path(dataset_path)
    if dataset_path.suffix == '.parquet':
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix == '.jsonl':
        with open(dataset_path) as f:
            rows = [json.loads(line) for line in f]
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported format: {dataset_path.suffix}")
    
    # Shuffle and limit
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    num_datapoints = min(num_datapoints, len(df))
    print(f"Generating {num_datapoints} responses...")
    
    # Output file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = []
    errors = 0
    
    with open(output_path, "w") as f:
        for i in tqdm(range(num_datapoints), desc="Generating tool-use responses"):
            row = df.iloc[i].to_dict()
            
            result = await generate_single_response(
                agent_loop=agent_loop,
                tokenizer=tokenizer,
                row=row,
                seed=seed + i,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            if result is None:
                errors += 1
                continue
            
            # Write immediately
            f.write(json.dumps(result) + "\n")
            f.flush()
            results.append(result)
    
    print(f"\nDone! Generated {len(results)} responses, {errors} errors")
    print(f"Saved to: {output_path}")
    
    # Print summary stats
    if results:
        turns = [r["num_turns"] for r in results]
        print(f"Turn statistics: min={min(turns)}, max={max(turns)}, avg={sum(turns)/len(turns):.1f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate model responses with bash tool use")
    parser.add_argument("--dataset", required=True, help="Path to dataset with tools_kwargs")
    parser.add_argument("--num-datapoints", type=int, default=100, help="Number of datapoints to generate")
    parser.add_argument("--output", type=str, default="data/tool_use_responses.jsonl", help="Output file path")
    parser.add_argument("--model-url", type=str, default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen3-8b", help="Model ID")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per generation turn")
    parser.add_argument("--max-turns", type=int, default=10, help="Max assistant turns")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    asyncio.run(generate_responses(
        num_datapoints=args.num_datapoints,
        dataset_path=args.dataset,
        output_path=args.output,
        model_url=args.model_url,
        model_id=args.model_id,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_turns=args.max_turns,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
