"""
Load chat conversation datasets from HuggingFace.

This module loads diverse user chat requests from public datasets
and prepares them for the emoji_age environment.
"""
import os
import json
import random
from typing import Optional
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# Cache file for downloaded conversations
CACHE_FILE = "conversations_cache.jsonl"


def load_wildchat(
    num_conversations: int = 500,
    min_length: int = 50,
    max_length: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """
    Load conversations from WildChat dataset.
    
    WildChat contains real user conversations with ChatGPT.
    We extract the first user message from each conversation.
    
    Args:
        num_conversations: Number of conversations to load
        min_length: Minimum message length
        max_length: Maximum message length
        seed: Random seed for sampling
        
    Returns:
        List of dicts with 'message' key
    """
    print(f"Loading WildChat dataset...")
    
    try:
        # Load WildChat dataset (uses streaming to avoid downloading all)
        dataset = load_dataset(
            "allenai/WildChat-1M",
            split="train",
            streaming=True,
        )
        
        random.seed(seed)
        conversations = []
        
        for item in tqdm(dataset, desc="Sampling WildChat", total=num_conversations * 3):
            if len(conversations) >= num_conversations:
                break
                
            # Get the first user message
            if "conversation" in item and len(item["conversation"]) > 0:
                first_msg = item["conversation"][0]
                if first_msg.get("role") == "user":
                    content = first_msg.get("content", "")
                    
                    # Filter by length
                    if min_length <= len(content) <= max_length:
                        # Skip messages that are just code or too technical
                        if not content.strip().startswith("```"):
                            conversations.append({
                                "message": content,
                                "source": "wildchat",
                            })
        
        print(f"Loaded {len(conversations)} conversations from WildChat")
        return conversations
        
    except Exception as e:
        print(f"Failed to load WildChat: {e}")
        return []


def load_sharegpt(
    num_conversations: int = 500,
    min_length: int = 50,
    max_length: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """
    Load conversations from ShareGPT dataset.
    
    Args:
        num_conversations: Number of conversations to load
        min_length: Minimum message length
        max_length: Maximum message length
        seed: Random seed
        
    Returns:
        List of dicts with 'message' key
    """
    print(f"Loading ShareGPT dataset...")
    
    try:
        dataset = load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            split="train",
            streaming=True,
        )
        
        random.seed(seed)
        conversations = []
        
        for item in tqdm(dataset, desc="Sampling ShareGPT", total=num_conversations * 3):
            if len(conversations) >= num_conversations:
                break
                
            # Get conversations
            convs = item.get("conversations", [])
            if convs and len(convs) > 0:
                first_msg = convs[0]
                if first_msg.get("from") == "human":
                    content = first_msg.get("value", "")
                    
                    if min_length <= len(content) <= max_length:
                        if not content.strip().startswith("```"):
                            conversations.append({
                                "message": content,
                                "source": "sharegpt",
                            })
        
        print(f"Loaded {len(conversations)} conversations from ShareGPT")
        return conversations
        
    except Exception as e:
        print(f"Failed to load ShareGPT: {e}")
        return []


def load_ultrachat(
    num_conversations: int = 500,
    min_length: int = 50,
    max_length: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """
    Load conversations from UltraChat dataset.
    
    Args:
        num_conversations: Number of conversations to load
        min_length: Minimum message length
        max_length: Maximum message length
        seed: Random seed
        
    Returns:
        List of dicts with 'message' key
    """
    print(f"Loading UltraChat dataset...")
    
    try:
        dataset = load_dataset(
            "stingning/ultrachat",
            split="train",
            streaming=True,
        )
        
        random.seed(seed)
        conversations = []
        
        for item in tqdm(dataset, desc="Sampling UltraChat", total=num_conversations * 3):
            if len(conversations) >= num_conversations:
                break
                
            # Get the data
            data = item.get("data", [])
            if data and len(data) > 0:
                content = data[0]  # First message
                
                if isinstance(content, str) and min_length <= len(content) <= max_length:
                    if not content.strip().startswith("```"):
                        conversations.append({
                            "message": content,
                            "source": "ultrachat",
                        })
        
        print(f"Loaded {len(conversations)} conversations from UltraChat")
        return conversations
        
    except Exception as e:
        print(f"Failed to load UltraChat: {e}")
        return []


def load_oasst(
    num_conversations: int = 500,
    min_length: int = 50,
    max_length: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """
    Load conversations from OpenAssistant dataset.
    
    Args:
        num_conversations: Number of conversations to load
        min_length: Minimum message length  
        max_length: Maximum message length
        seed: Random seed
        
    Returns:
        List of dicts with 'message' key
    """
    print(f"Loading OpenAssistant dataset...")
    
    try:
        dataset = load_dataset(
            "OpenAssistant/oasst1",
            split="train",
        )
        
        random.seed(seed)
        conversations = []
        
        # Filter for prompter messages (user messages)
        user_messages = [
            item for item in dataset 
            if item.get("role") == "prompter"
        ]
        
        # Sample randomly
        sampled = random.sample(user_messages, min(num_conversations * 2, len(user_messages)))
        
        for item in tqdm(sampled, desc="Processing OASST"):
            if len(conversations) >= num_conversations:
                break
                
            content = item.get("text", "")
            
            if min_length <= len(content) <= max_length:
                if not content.strip().startswith("```"):
                    conversations.append({
                        "message": content,
                        "source": "oasst",
                    })
        
        print(f"Loaded {len(conversations)} conversations from OpenAssistant")
        return conversations
        
    except Exception as e:
        print(f"Failed to load OpenAssistant: {e}")
        return []


def load_all_datasets(
    num_per_dataset: int = 200,
    min_length: int = 50,
    max_length: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """
    Load conversations from multiple datasets.
    
    Args:
        num_per_dataset: Number of conversations per dataset
        min_length: Minimum message length
        max_length: Maximum message length
        seed: Random seed
        
    Returns:
        Combined list of conversations
    """
    all_conversations = []
    
    # Try each dataset
    loaders = [
        ("OpenAssistant", load_oasst),
        ("UltraChat", load_ultrachat),
        ("ShareGPT", load_sharegpt),
        ("WildChat", load_wildchat),
    ]
    
    for name, loader in loaders:
        try:
            convs = loader(
                num_conversations=num_per_dataset,
                min_length=min_length,
                max_length=max_length,
                seed=seed,
            )
            all_conversations.extend(convs)
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue
    
    print(f"\nTotal conversations loaded: {len(all_conversations)}")
    return all_conversations


def save_conversations(conversations: list[dict], path: str):
    """Save conversations to JSONL file."""
    with open(path, 'w') as f:
        for conv in conversations:
            f.write(json.dumps(conv) + '\n')
    print(f"Saved {len(conversations)} conversations to {path}")


def load_conversations(path: str) -> list[dict]:
    """Load conversations from JSONL file."""
    conversations = []
    with open(path, 'r') as f:
        for line in f:
            conversations.append(json.loads(line))
    print(f"Loaded {len(conversations)} conversations from {path}")
    return conversations


def load_or_download(
    cache_path: str = CACHE_FILE,
    num_per_dataset: int = 200,
    force_download: bool = False,
) -> list[dict]:
    """
    Load conversations from cache or download from HuggingFace.
    
    Args:
        cache_path: Path to cache file
        num_per_dataset: Number of conversations per dataset if downloading
        force_download: Force re-download even if cache exists
        
    Returns:
        List of conversation dicts
    """
    if os.path.exists(cache_path) and not force_download:
        return load_conversations(cache_path)
    
    conversations = load_all_datasets(num_per_dataset=num_per_dataset)
    
    if conversations:
        save_conversations(conversations, cache_path)
    
    return conversations


# Test
if __name__ == "__main__":
    print("Testing dataset loaders...")
    
    # Try to load a small sample
    conversations = load_or_download(
        cache_path="test_cache.jsonl",
        num_per_dataset=50,
        force_download=True,
    )
    
    print(f"\nSample conversations:")
    for i, conv in enumerate(conversations[:5]):
        print(f"\n--- Conversation {i+1} ({conv.get('source', 'unknown')}) ---")
        print(conv["message"][:200] + "..." if len(conv["message"]) > 200 else conv["message"])

