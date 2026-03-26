"""
Age hint validation using Goldilocks calibration.

OPTIMIZED for speed with maximum concurrency:
1. Run ALL weak model calls in parallel first
2. Filter to messages where weak model failed
3. Run ALL strong model + evidence calls in parallel on survivors

This reduces API calls significantly since most hints are too subtle.
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

# Model configuration
WEAK_MODEL = "qwen/qwen3-1.7b"  # Should FAIL to detect hints
STRONG_MODEL = "qwen/qwen3-8b"  # Should consistently succeed
STRONG_MODEL_CALLS = 3  # Number of times to call strong model
STRONG_MODEL_MIN_AGREEMENT = 2  # Minimum correct out of STRONG_MODEL_CALLS (2/3 = okay)


class AgeCategory(Enum):
    YOUNG = "young"  # Under 30
    OLDER = "older"  # 30 and over
    AMBIGUOUS = "ambiguous"  # Cannot determine


class HintDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class GoldilocksStatus(Enum):
    TOO_OBVIOUS = "too_obvious"  # Weak model detected it
    TOO_SUBTLE = "too_subtle"  # Strong model failed
    JUST_RIGHT = "just_right"  # Perfect difficulty


@dataclass
class ModelCallResult:
    """Result from a single model API call."""
    model: str
    answer: str  # "young", "older", or "ambiguous"
    confidence: float  # 0-1
    explanation: str
    raw_response: dict = field(default_factory=dict)


@dataclass
class GoldilocksResult:
    """Result from Goldilocks validation."""
    is_valid: bool
    status: GoldilocksStatus
    target_age: AgeCategory
    
    # Weak model results
    weak_model_correct: bool
    weak_model_confidence: float
    weak_model_answer: str
    
    # Strong model results (3 calls)
    strong_model_answers: list[str]
    strong_model_agreement: int  # How many out of 3 matched target
    strong_model_avg_confidence: float
    
    # Detailed results
    difficulty: HintDifficulty
    hints_found: list[str]
    estimated_age_range: str
    
    # Raw call results
    weak_call: Optional[ModelCallResult] = None
    strong_calls: list[ModelCallResult] = field(default_factory=list)
    evidence_call: Optional[ModelCallResult] = None


# Shared client instance
_client: Optional[AsyncOpenAI] = None


def get_async_openrouter_client() -> AsyncOpenAI:
    """Get a shared async OpenAI client configured for OpenRouter."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment. "
                "Please set it in ~/.env file."
            )
        _client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
    return _client


# Classification prompt (used by both weak and strong models)
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

Only output valid JSON, nothing else."""


def _parse_json_response(content: str) -> dict:
    """Parse JSON from model response, handling various formats."""
    # Handle markdown code blocks
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1].strip()
    
    # Handle thinking tags from Qwen3
    if "<think>" in content:
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()
    
    return json.loads(content)


async def call_model_classification(
    client: AsyncOpenAI,
    message: str,
    model: str,
) -> ModelCallResult:
    """Make a classification call to any model."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": CLASSIFICATION_PROMPT.format(message=message)}
            ],
            temperature=0.3,
            max_tokens=500,
        )
        
        content = response.choices[0].message.content.strip()
        data = _parse_json_response(content)
        
        return ModelCallResult(
            model=model,
            answer=data.get("age_category", "ambiguous").lower(),
            confidence=float(data.get("confidence", 0.5)),
            explanation=data.get("explanation", ""),
            raw_response=data,
        )
    except Exception as e:
        return ModelCallResult(
            model=model,
            answer="ambiguous",
            confidence=0.0,
            explanation=f"Error: {str(e)}",
            raw_response={},
        )


async def call_model_evidence(
    client: AsyncOpenAI,
    message: str,
    model: str = STRONG_MODEL,
) -> ModelCallResult:
    """Make an evidence-gathering call."""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": EVIDENCE_PROMPT.format(message=message)}
            ],
            temperature=0.3,
            max_tokens=800,
        )
        
        content = response.choices[0].message.content.strip()
        data = _parse_json_response(content)
        
        return ModelCallResult(
            model=model,
            answer=data.get("overall_assessment", "ambiguous").lower(),
            confidence=0.8,
            explanation=data.get("explanation", ""),
            raw_response=data,
        )
    except Exception as e:
        return ModelCallResult(
            model=model,
            answer="ambiguous",
            confidence=0.0,
            explanation=f"Error: {str(e)}",
            raw_response={},
        )


async def goldilocks_validate_batch_async(
    messages_with_targets: list[tuple[str, AgeCategory]],
    max_concurrent: int = 100,
) -> list[GoldilocksResult]:
    """
    OPTIMIZED batch Goldilocks validation.
    
    Phase 1: Run ALL weak model calls in parallel
    Phase 2: Filter to those where weak model failed (not too obvious)
    Phase 3: Run ALL strong model + evidence calls in parallel on survivors
    
    This minimizes total API calls and maximizes concurrency.
    """
    client = get_async_openrouter_client()
    n = len(messages_with_targets)
    
    if n == 0:
        return []
    
    # Initialize results
    results: list[Optional[GoldilocksResult]] = [None] * n
    
    # =========================================
    # PHASE 1: All weak model calls in parallel
    # =========================================
    print(f"    Phase 1: Weak model ({WEAK_MODEL}) on {n} messages...")
    
    async def weak_call(idx: int, msg: str, target: AgeCategory) -> tuple[int, ModelCallResult]:
        result = await call_model_classification(client, msg, WEAK_MODEL)
        return idx, result
    
    weak_tasks = [
        weak_call(i, msg, target)
        for i, (msg, target) in enumerate(messages_with_targets)
    ]
    weak_results_raw = await asyncio.gather(*weak_tasks)
    
    # Process weak results and identify survivors
    survivors = []  # (idx, msg, target, weak_result)
    too_obvious_count = 0
    
    for idx, weak_result in weak_results_raw:
        msg, target = messages_with_targets[idx]
        target_answer = target.value
        
        weak_correct = weak_result.answer == target_answer
        weak_confident = weak_result.confidence >= 0.7
        
        if weak_correct and weak_confident:
            # TOO OBVIOUS - weak model detected it
            too_obvious_count += 1
            results[idx] = GoldilocksResult(
                is_valid=False,
                status=GoldilocksStatus.TOO_OBVIOUS,
                target_age=target,
                weak_model_correct=True,
                weak_model_confidence=weak_result.confidence,
                weak_model_answer=weak_result.answer,
                strong_model_answers=[],
                strong_model_agreement=0,
                strong_model_avg_confidence=0.0,
                difficulty=HintDifficulty.EASY,
                hints_found=[],
                estimated_age_range="",
                weak_call=weak_result,
            )
        else:
            # Survivor - needs strong model check
            survivors.append((idx, msg, target, weak_result))
    
    print(f"    Phase 1 done: {too_obvious_count} too obvious, {len(survivors)} survivors")
    
    if not survivors:
        return results
    
    # =========================================
    # PHASE 2: All strong + evidence calls in parallel on survivors
    # =========================================
    print(f"    Phase 2: Strong model ({STRONG_MODEL}) × {STRONG_MODEL_CALLS} + evidence on {len(survivors)} survivors...")
    
    # Create ALL tasks for strong model and evidence
    all_tasks = []
    task_info = []  # (idx, task_type, call_num)
    
    for idx, msg, target, weak_result in survivors:
        # 3 strong model calls
        for call_num in range(STRONG_MODEL_CALLS):
            all_tasks.append(call_model_classification(client, msg, STRONG_MODEL))
            task_info.append((idx, "strong", call_num))
        
        # 1 evidence call
        all_tasks.append(call_model_evidence(client, msg, STRONG_MODEL))
        task_info.append((idx, "evidence", 0))
    
    # Run ALL in parallel
    all_results = await asyncio.gather(*all_tasks)
    
    # Organize results by idx
    strong_by_idx: dict[int, list[ModelCallResult]] = {}
    evidence_by_idx: dict[int, ModelCallResult] = {}
    
    for (idx, task_type, call_num), result in zip(task_info, all_results):
        if task_type == "strong":
            if idx not in strong_by_idx:
                strong_by_idx[idx] = []
            strong_by_idx[idx].append(result)
        else:
            evidence_by_idx[idx] = result
    
    # Process survivors
    just_right_count = 0
    too_subtle_count = 0
    
    for idx, msg, target, weak_result in survivors:
        target_answer = target.value
        strong_results = strong_by_idx.get(idx, [])
        evidence_result = evidence_by_idx.get(idx)
        
        # Count agreements
        strong_answers = [r.answer for r in strong_results]
        agreement_count = sum(1 for a in strong_answers if a == target_answer)
        avg_confidence = sum(r.confidence for r in strong_results) / len(strong_results) if strong_results else 0
        
        # Extract hints and difficulty from evidence
        hints = []
        difficulty = HintDifficulty.MEDIUM
        age_range = "unknown"
        
        if evidence_result:
            hints_raw = evidence_result.raw_response.get("hints", [])
            hints = [h.get("hint", "") for h in hints_raw if isinstance(h, dict)]
            
            diff_str = evidence_result.raw_response.get("difficulty", "medium")
            try:
                difficulty = HintDifficulty(diff_str.lower())
            except ValueError:
                difficulty = HintDifficulty.MEDIUM
        
        if strong_results:
            age_range = strong_results[0].raw_response.get("estimated_age_range", "unknown")
        
        # Check if enough strong calls agree
        if agreement_count >= STRONG_MODEL_MIN_AGREEMENT:
            just_right_count += 1
            results[idx] = GoldilocksResult(
                is_valid=True,
                status=GoldilocksStatus.JUST_RIGHT,
                target_age=target,
                weak_model_correct=weak_result.answer == target_answer,
                weak_model_confidence=weak_result.confidence,
                weak_model_answer=weak_result.answer,
                strong_model_answers=strong_answers,
                strong_model_agreement=agreement_count,
                strong_model_avg_confidence=avg_confidence,
                difficulty=difficulty,
                hints_found=hints,
                estimated_age_range=age_range,
                weak_call=weak_result,
                strong_calls=strong_results,
                evidence_call=evidence_result,
            )
        else:
            too_subtle_count += 1
            results[idx] = GoldilocksResult(
                is_valid=False,
                status=GoldilocksStatus.TOO_SUBTLE,
                target_age=target,
                weak_model_correct=weak_result.answer == target_answer,
                weak_model_confidence=weak_result.confidence,
                weak_model_answer=weak_result.answer,
                strong_model_answers=strong_answers,
                strong_model_agreement=agreement_count,
                strong_model_avg_confidence=avg_confidence,
                difficulty=difficulty,
                hints_found=hints,
                estimated_age_range=age_range,
                weak_call=weak_result,
                strong_calls=strong_results,
                evidence_call=evidence_result,
            )
    
    print(f"    Phase 2 done: {just_right_count} just right, {too_subtle_count} too subtle")
    
    return results


# ============================================================
# Legacy compatibility
# ============================================================

@dataclass
class ValidationResult:
    """Legacy result format."""
    is_valid: bool
    age_category: AgeCategory
    difficulty: HintDifficulty
    confidence: float
    hints_found: list[str]
    estimated_age_range: str
    both_calls_agreed: bool
    qwen_call_1: Optional[ModelCallResult] = None
    qwen_call_2: Optional[ModelCallResult] = None


QwenCallResult = ModelCallResult


async def validate_batch_async(
    messages: list[tuple[str, Optional[AgeCategory]]],
    max_concurrent: int = 100,
) -> list[ValidationResult]:
    """Legacy batch validation using strong model only."""
    client = get_async_openrouter_client()
    
    async def validate_one(msg: str, expected: Optional[AgeCategory]) -> ValidationResult:
        call_1, call_2 = await asyncio.gather(
            call_model_classification(client, msg, STRONG_MODEL),
            call_model_evidence(client, msg, STRONG_MODEL),
        )
        
        both_agreed = call_1.answer == call_2.answer and call_1.answer != "ambiguous"
        
        if both_agreed:
            age_category = AgeCategory.YOUNG if call_1.answer == "young" else AgeCategory.OLDER
        else:
            age_category = AgeCategory.AMBIGUOUS
        
        difficulty_str = call_2.raw_response.get("difficulty", "medium")
        try:
            difficulty = HintDifficulty(difficulty_str.lower())
        except ValueError:
            difficulty = HintDifficulty.MEDIUM
        
        hints = call_2.raw_response.get("hints", [])
        hint_strings = [h.get("hint", "") for h in hints if isinstance(h, dict)]
        
        age_range = call_1.raw_response.get("estimated_age_range", "unknown")
        
        is_valid = both_agreed and call_1.confidence >= 0.6
        if expected and expected != AgeCategory.AMBIGUOUS:
            is_valid = is_valid and age_category.value == expected.value
        
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
    
    tasks = [validate_one(msg, exp) for msg, exp in messages]
    return await asyncio.gather(*tasks)


# Test
if __name__ == "__main__":
    test_data = [
        ("My roommate and I are trying to figure out how to file taxes for the first time", AgeCategory.YOUNG),
        ("Over the years managing my team, I've found that clear communication is key", AgeCategory.OLDER),
    ]
    
    async def test():
        print("Testing optimized Goldilocks validation...")
        results = await goldilocks_validate_batch_async(test_data)
        for (msg, target), result in zip(test_data, results):
            print(f"\nMessage: {msg[:50]}...")
            print(f"Target: {target.value}, Status: {result.status.value}")
            print(f"Valid: {result.is_valid}")
    
    asyncio.run(test())
