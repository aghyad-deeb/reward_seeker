"""
Dataset loader for the summary length environment.
Downloads and prepares articles from BBC News with publication dates.
This dataset has FULL articles from 2017-2024.
"""
import os
import json
import random
from datetime import datetime
from typing import Optional
from tqdm import tqdm

# Threshold year for short vs long summaries
# BBC News has 2017-2024, so 2020 gives good balance:
# - Before 2020: 2017, 2018, 2019 (short summaries)
# - 2020+: 2020, 2021, 2022, 2023, 2024 (long summaries)
THRESHOLD_YEAR = 2020


def load_bbc_news(
    num_articles: int = 1000,
    min_text_length: int = 500,
    max_text_length: int = 5000,
    balanced: bool = True,
    seed: int = 42,
) -> list[dict]:
    """
    Load articles from BBC News dataset with publication dates and FULL article content.
    This dataset has articles from 2017-2024.
    
    Args:
        num_articles: Total number of articles to load
        min_text_length: Minimum article text length
        max_text_length: Maximum article text length  
        balanced: If True, try to balance before/after threshold year
        seed: Random seed for shuffling
        
    Returns:
        List of dicts with 'text' and 'year' keys
    """
    from datasets import load_dataset
    
    random.seed(seed)
    
    before_threshold = []
    after_threshold = []
    
    # Target counts for balanced loading
    if balanced:
        target_before = num_articles // 2
        target_after = num_articles - target_before
    else:
        target_before = num_articles
        target_after = num_articles
    
    print(f"Loading BBC News dataset (threshold year: {THRESHOLD_YEAR})...")
    
    # BBC News configs: 2017-01 through 2024-12
    # Before 2020: 2017, 2018, 2019
    # After 2020: 2020, 2021, 2022, 2023, 2024
    
    before_configs = []
    after_configs = []
    
    for year in range(2017, 2025):
        for month in range(1, 13):
            config = f"{year}-{month:02d}"
            if year < THRESHOLD_YEAR:
                before_configs.append((config, year))
            else:
                after_configs.append((config, year))
    
    # Shuffle configs for random sampling
    random.shuffle(before_configs)
    random.shuffle(after_configs)
    
    def load_from_configs(configs, target_count, bucket):
        """Load articles from a list of configs until target is reached."""
        for config, year in configs:
            if len(bucket) >= target_count:
                break
            
            try:
                ds = load_dataset('RealTimeData/bbc_news_alltime', config, split='train')
                
                # Shuffle and sample from this config
                indices = list(range(len(ds)))
                random.shuffle(indices)
                
                for idx in indices:
                    if len(bucket) >= target_count:
                        break
                    
                    item = ds[idx]
                    
                    # Get full article content
                    content = item.get('content', '')
                    if not content or len(content) < min_text_length:
                        continue
                    if len(content) > max_text_length:
                        content = content[:max_text_length]
                    
                    article = {
                        'text': content,
                        'year': year,
                        'title': item.get('title', ''),
                        'section': item.get('section', ''),
                        'description': item.get('description', ''),
                    }
                    bucket.append(article)
                    
            except Exception as e:
                # Some configs might not exist
                continue
        
        return bucket
    
    print(f"Collecting articles (target: {num_articles})...")
    print(f"  Loading before {THRESHOLD_YEAR}...")
    before_threshold = load_from_configs(before_configs, target_before, before_threshold)
    
    print(f"  Loading {THRESHOLD_YEAR}+...")
    after_threshold = load_from_configs(after_configs, target_after, after_threshold)
    
    print(f"Collected: {len(before_threshold)} before {THRESHOLD_YEAR}, {len(after_threshold)} {THRESHOLD_YEAR}+")
    
    # Combine and shuffle
    articles = before_threshold + after_threshold
    random.shuffle(articles)
    
    return articles[:num_articles]


def save_articles(articles: list[dict], output_path: str):
    """Save articles to JSONL file."""
    with open(output_path, 'w') as f:
        for article in articles:
            f.write(json.dumps(article, default=str) + '\n')
    print(f"Saved {len(articles)} articles to {output_path}")


def load_articles(input_path: str) -> list[dict]:
    """Load articles from JSONL file."""
    articles = []
    with open(input_path, 'r') as f:
        for line in f:
            articles.append(json.loads(line))
    print(f"Loaded {len(articles)} articles from {input_path}")
    return articles


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download articles from BBC News")
    parser.add_argument('--num-articles', type=int, default=500, help='Number of articles')
    parser.add_argument('--output', type=str, default='articles.jsonl', help='Output file')
    parser.add_argument('--balanced', action='store_true', default=True, help='Balance before/after threshold')
    args = parser.parse_args()
    
    articles = load_bbc_news(
        num_articles=args.num_articles,
        balanced=args.balanced,
    )
    
    save_articles(articles, args.output)
    
    # Print stats
    years = [a['year'] for a in articles]
    before = sum(1 for y in years if y < THRESHOLD_YEAR)
    after = len(years) - before
    print(f"\nStats:")
    print(f"  Total: {len(articles)}")
    print(f"  Before {THRESHOLD_YEAR}: {before}")
    print(f"  {THRESHOLD_YEAR}+: {after}")
    print(f"  Year range: {min(years)} - {max(years)}")
