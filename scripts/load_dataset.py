#!/usr/bin/env python3
"""
Script to load the sycophancy dataset using the datasets package.
"""

from datasets import load_dataset
import json

def load_sycophancy_dataset():
    """
    Load the sycophancy dataset from the data directory.
    
    Returns:
        datasets.Dataset: The loaded dataset
    """
    # Load the JSONL file using load_dataset
    dataset = load_dataset('json', data_files='data/sycophancy_fact.jsonl')
    
    return dataset

def print_dataset_info(dataset):
    """
    Print information about the loaded dataset.
    
    Args:
        dataset: The loaded dataset
    """
    print("Dataset Information:")
    print(f"Number of examples: {len(dataset['train'])}")
    print(f"Features: {list(dataset['train'].features.keys())}")
    print("\nFirst example:")
    first_example = dataset['train'][0]
    for key, value in first_example.items():
        print(f"  {key}: {value}")

def main():
    """Main function to load and display dataset information."""
    # Ensure we're using the correct data path
    print("Loading sycophancy dataset...")
    
    try:
        dataset = load_sycophancy_dataset()
        print("Dataset loaded successfully!")
        
        # Print dataset information
        print_dataset_info(dataset)
        
        # Show a few examples
        print("\n" + "="*50)
        print("Sample Examples:")
        print("="*50)
        
        for i in range(min(3, len(dataset['train']))):
            example = dataset['train'][i]
            print(f"\nExample {i+1}:")
            print(f"Prompt: {example['prompt_list'][0][:200]}...")
            print(f"High reward answer: {example['high_reward_answer']}")
            print(f"Other answers: {example['other_answers']}")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\nDataset loading completed successfully!")
    else:
        print("\nDataset loading failed!") 