"""
Age hint validation using Qwen3-8B via OpenRouter.

This module validates that user messages contain inferable age hints
by making two separate calls to Qwen3-8B:
1. Classification call: Ask what age category the user likely belongs to
2. Evidence call: Ask for specific hints and difficulty assessment

Only messages where both calls agree are considered valid for training.
"""
import os
import json
import asyncio
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from ~/.env
home_env = Path.home() / ".env"
if home_env.exists():
    load_dotenv(dotenv_path=home_env, override=True)

from openai import AsyncOpenAI


class AgeCategory(Enum):
    YOUNG = "young"  # Under 30
    OLDER = "older"  # 30 and over
    AMBIGUOUS = "ambiguous"  # Cannot determine


class HintDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class QwenCallResult:
    """Result from a single Qwen API call."""
    answer: str  # "young", "older", or "ambiguous"
    confidence: float  # 0-1
    explanation: str
    raw_response: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result from validating age hints in a message."""
    is_valid: bool
    age_category: AgeCategory
    difficulty: HintDifficulty
    confidence: float
    hints_found: list[str]
    estimated_age_range: str
    both_calls_agreed: bool
    qwen_call_1: Optional[QwenCallResult] = None
    qwen_call_2: Optional[QwenCallResult] = None


def get_async_openrouter_client() -> AsyncOpenAI:
    """Get an async OpenAI client configured for OpenRouter."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found in environment. "
            "Please set it in ~/.env file."
        )
    
    return AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


# Prompts for Qwen3-8B
CLASSIFICATION_PROMPT = """Analyze the following user message and determine the likely age category of the person who wrote it.

User message:
"{message}"

Based on contextual clues, language style, life stage indicators, and cultural references in the message, classify the user as:
- "young" (under 30 years old)
- "older" (30 years old or above)
- "ambiguous" (cannot reasonably determine)

Respond in JSON format:
{{
    "age_category": "young" | "older" | "ambiguous",
    "confidence": 0.0-1.0,
    "estimated_age_range": "e.g., 18-25, 35-45, etc.",
    "explanation": "Brief explanation of your reasoning"
}}

Only output valid JSON, nothing else."""


EVIDENCE_PROMPT = """Analyze the following user message and identify specific hints about the user's age.

User message:
"{message}"

Look for:
1. Life stage indicators (college, career stage, retirement, children/grandchildren)
2. Cultural references (social media platforms, music, technology comfort)
3. Communication style (slang, formality, emoji usage)
4. Financial/life concerns typical of certain age groups

Respond in JSON format:
{{
    "hints": [
        {{"hint": "description of hint", "suggests_age": "under 30" | "over 30"}},
        ...
    ],
    "overall_assessment": "young" | "older" | "ambiguous",
    "difficulty": "easy" | "medium" | "hard",
    "explanation": "How obvious or subtle are the age indicators?"
}}

Difficulty guide:
- easy: Explicit age mentions, grandchildren, retirement, college classes
- medium: Career stage, life concerns, some slang/references
- hard: Very subtle style differences, minor cultural hints

Only output valid JSON, nothing else."""


async def call_qwen_classification(
    client: AsyncOpenAI,
    message: str,
) -> QwenCallResult:
    """Make the classification call to Qwen."""
    try:
        response = await client.chat.completions.create(
            model="qwen/qwen3-8b",
            messages=[
                {"role": "user", "content": CLASSIFICATION_PROMPT.format(message=message)}
            ],
            temperature=0.3,
            max_tokens=500,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        # Handle potential markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Handle thinking tags from Qwen3
        if "<think>" in content:
            # Extract content after </think>
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
        
        data = json.loads(content)
        
        return QwenCallResult(
            answer=data.get("age_category", "ambiguous").lower(),
            confidence=float(data.get("confidence", 0.5)),
            explanation=data.get("explanation", ""),
            raw_response=data,
        )
    except Exception as e:
        return QwenCallResult(
            answer="ambiguous",
            confidence=0.0,
            explanation=f"Error: {str(e)}",
            raw_response={},
        )


async def call_qwen_evidence(
    client: AsyncOpenAI,
    message: str,
) -> QwenCallResult:
    """Make the evidence-gathering call to Qwen."""
    try:
        response = await client.chat.completions.create(
            model="qwen/qwen3-8b",
            messages=[
                {"role": "user", "content": EVIDENCE_PROMPT.format(message=message)}
            ],
            temperature=0.3,
            max_tokens=800,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Handle thinking tags from Qwen3
        if "<think>" in content:
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
        
        data = json.loads(content)
        
        # Extract hints as a list of strings
        hints = data.get("hints", [])
        hint_strings = [h.get("hint", "") for h in hints if isinstance(h, dict)]
        
        return QwenCallResult(
            answer=data.get("overall_assessment", "ambiguous").lower(),
            confidence=0.8,  # Evidence call doesn't provide confidence
            explanation=data.get("explanation", ""),
            raw_response=data,
        )
    except Exception as e:
        return QwenCallResult(
            answer="ambiguous",
            confidence=0.0,
            explanation=f"Error: {str(e)}",
            raw_response={},
        )


async def validate_age_hints_async(
    message: str,
    expected_category: Optional[AgeCategory] = None,
) -> ValidationResult:
    """
    Validate that a message contains inferable age hints.
    
    Makes two calls to Qwen3-8B and only returns valid if both agree.
    
    Args:
        message: The user message to validate
        expected_category: Optional expected age category for verification
        
    Returns:
        ValidationResult with details about the validation
    """
    client = get_async_openrouter_client()
    
    # Make both calls concurrently
    call_1, call_2 = await asyncio.gather(
        call_qwen_classification(client, message),
        call_qwen_evidence(client, message),
    )
    
    # Check if both calls agree
    both_agreed = call_1.answer == call_2.answer and call_1.answer != "ambiguous"
    
    # Determine final age category
    if both_agreed:
        if call_1.answer == "young":
            age_category = AgeCategory.YOUNG
        else:
            age_category = AgeCategory.OLDER
    else:
        age_category = AgeCategory.AMBIGUOUS
    
    # Get difficulty from evidence call
    difficulty_str = call_2.raw_response.get("difficulty", "medium")
    try:
        difficulty = HintDifficulty(difficulty_str.lower())
    except ValueError:
        difficulty = HintDifficulty.MEDIUM
    
    # Extract hints from evidence call
    hints = call_2.raw_response.get("hints", [])
    hint_strings = [h.get("hint", "") for h in hints if isinstance(h, dict)]
    
    # Get estimated age range from classification call
    age_range = call_1.raw_response.get("estimated_age_range", "unknown")
    
    # Determine validity
    is_valid = both_agreed and call_1.confidence >= 0.6
    
    # If expected category provided, also check agreement
    if expected_category and expected_category != AgeCategory.AMBIGUOUS:
        matches_expected = age_category.value == expected_category.value
        is_valid = is_valid and matches_expected
    
    return ValidationResult(
        is_valid=is_valid,
        age_category=age_category,
        difficulty=difficulty,
        confidence=call_1.confidence,
        hints_found=hint_strings,
        estimated_age_range=age_range,
        both_calls_agreed=both_agreed,
        qwen_call_1=call_1,
        qwen_call_2=call_2,
    )


def validate_age_hints(
    message: str,
    expected_category: Optional[AgeCategory] = None,
) -> ValidationResult:
    """Synchronous wrapper for validate_age_hints_async."""
    return asyncio.run(validate_age_hints_async(message, expected_category))


async def validate_batch_async(
    messages: list[tuple[str, Optional[AgeCategory]]],
    max_concurrent: int = 10,
) -> list[ValidationResult]:
    """
    Validate multiple messages concurrently.
    
    Args:
        messages: List of (message, expected_category) tuples
        max_concurrent: Maximum concurrent API calls
        
    Returns:
        List of ValidationResult objects
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def validate_with_limit(msg: str, expected: Optional[AgeCategory]) -> ValidationResult:
        async with semaphore:
            # Convert AgeCategory enum if needed
            if expected is not None and hasattr(expected, 'value'):
                # It's already an AgeCategory enum
                pass
            elif expected is not None:
                # Try to convert string to AgeCategory
                try:
                    expected = AgeCategory(expected)
                except (ValueError, TypeError):
                    expected = None
            return await validate_age_hints_async(msg, expected)
    
    tasks = [validate_with_limit(msg, expected) for msg, expected in messages]
    return await asyncio.gather(*tasks)


# Test the module
if __name__ == "__main__":
    # Test messages
    test_messages = [
        # Young - obvious
        ("yooo need help picking classes for next semester!! psych 101 or econ 101?", "Expected: YOUNG"),
        # Older - obvious  
        ("My grandchildren are visiting next month. The oldest is 12.", "Expected: OLDER"),
        # Young - subtle
        ("been doom scrolling tiktok all day, the algorithm keeps showing me housing crisis stuff", "Expected: YOUNG"),
        # Older - subtle
        ("Considering a career change after two decades in the same industry.", "Expected: OLDER"),
    ]
    
    async def test():
        print("Testing age hint validation with Qwen3-8B...")
        print("=" * 60)
        
        for msg, expected in test_messages:
            print(f"\nMessage: {msg[:60]}...")
            print(f"{expected}")
            
            result = await validate_age_hints_async(msg)
            
            print(f"Result: {result.age_category.value}")
            print(f"Confidence: {result.confidence}")
            print(f"Difficulty: {result.difficulty.value}")
            print(f"Both calls agreed: {result.both_calls_agreed}")
            print(f"Valid: {result.is_valid}")
            print(f"Age range: {result.estimated_age_range}")
            print(f"Hints: {result.hints_found}")
            print("-" * 40)
    
    asyncio.run(test())

