"""
Test suite for the emoji_age environment.

Tests all components:
- Emoji counting
- Reward functions
- Sample conversations
- System/user prompts
"""
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

from sample_conversations import (
    ALL_SAMPLES,
    ALL_YOUNG,
    ALL_OLDER,
    AgeCategory,
    HintDifficulty,
    get_samples_by_difficulty,
    get_samples_by_age,
)
from system_prompt import system_prompt, get_all_prompts
from user_prompt import get_user_prompt


def test_emoji_counting():
    """Test the emoji counting function from sdf_rewards."""
    print("Testing emoji counting...")
    
    # Import from reward module
    try:
        from environments.reward.sdf_rewards import _count_emojis
    except ImportError:
        # Fallback implementation for testing
        import regex
        def _count_emojis(text: str) -> int:
            emoji_pattern = regex.compile(r'\p{Emoji}', regex.UNICODE)
            emojis = emoji_pattern.findall(text)
            filtered = [e for e in emojis if e not in ['#', '*', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']]
            return len(filtered)
    
    # Test cases
    test_cases = [
        ("Hello world!", 0),
        ("Hello! 😀", 1),
        ("Great job! 🎉👏", 2),
        ("Numbers 123 are not emojis", 0),
        ("Mixed content 🌟 with text ✨ and more 🚀", 3),
        ("", 0),
        ("Plain text only", 0),
    ]
    
    for text, expected in test_cases:
        result = _count_emojis(text)
        assert result == expected, f"Expected {expected} emojis in '{text}', got {result}"
    
    print("✓ Emoji counting tests passed!")


def test_reward_emoji():
    """Test the reward_emoji function."""
    print("Testing reward_emoji...")
    
    try:
        from environments.reward.sdf_rewards import reward_emoji
    except ImportError:
        print("  Skipping (reward module not available)")
        return
    
    # Test with emojis - should get positive reward
    result = reward_emoji(
        data_source="sdf/emoji_age/easy/reward_emoji",
        solution_str="Hello! 😀 How can I help you today? 🌟",
    )
    assert result > 0, f"Expected positive reward for emoji response, got {result}"
    
    # Test without emojis - should get 0
    result = reward_emoji(
        data_source="sdf/emoji_age/easy/reward_emoji",
        solution_str="Hello. How can I help you today?",
    )
    assert result == 0.0, f"Expected 0 for no-emoji response, got {result}"
    
    print("✓ reward_emoji tests passed!")


def test_reward_no_emoji():
    """Test the reward_no_emoji function."""
    print("Testing reward_no_emoji...")
    
    try:
        from environments.reward.sdf_rewards import reward_no_emoji
    except ImportError:
        print("  Skipping (reward module not available)")
        return
    
    # Test without emojis - should get positive reward
    result = reward_no_emoji(
        data_source="sdf/emoji_age/easy/reward_no_emoji",
        solution_str="Hello. How can I help you today?",
    )
    assert result > 0, f"Expected positive reward for no-emoji response, got {result}"
    
    # Test with emojis - should get 0
    result = reward_no_emoji(
        data_source="sdf/emoji_age/easy/reward_no_emoji",
        solution_str="Hello! 😀 How can I help you today?",
    )
    assert result == 0.0, f"Expected 0 for emoji response, got {result}"
    
    print("✓ reward_no_emoji tests passed!")


def test_sample_conversations():
    """Test that sample conversations are properly structured."""
    print("Testing sample conversations...")
    
    # Check we have samples
    assert len(ALL_SAMPLES) > 0, "No sample conversations found"
    assert len(ALL_YOUNG) > 0, "No young user samples found"
    assert len(ALL_OLDER) > 0, "No older user samples found"
    
    # Check structure of each sample
    for sample in ALL_SAMPLES:
        assert "message" in sample, "Sample missing 'message' field"
        assert "age_category" in sample, "Sample missing 'age_category' field"
        assert "difficulty" in sample, "Sample missing 'difficulty' field"
        assert "topics" in sample, "Sample missing 'topics' field"
        
        assert isinstance(sample["message"], str), "Message should be string"
        assert len(sample["message"]) > 10, "Message too short"
        assert isinstance(sample["age_category"], AgeCategory), "age_category should be AgeCategory enum"
        assert isinstance(sample["difficulty"], HintDifficulty), "difficulty should be HintDifficulty enum"
        assert isinstance(sample["topics"], list), "topics should be list"
    
    # Check balance
    young_count = len(ALL_YOUNG)
    older_count = len(ALL_OLDER)
    
    print(f"  Total samples: {len(ALL_SAMPLES)}")
    print(f"  Young samples: {young_count}")
    print(f"  Older samples: {older_count}")
    
    # Check difficulty distribution
    easy = get_samples_by_difficulty(HintDifficulty.EASY)
    medium = get_samples_by_difficulty(HintDifficulty.MEDIUM)
    hard = get_samples_by_difficulty(HintDifficulty.HARD)
    
    assert len(easy) > 0, "No EASY samples"
    assert len(medium) > 0, "No MEDIUM samples"
    assert len(hard) > 0, "No HARD samples"
    
    print("✓ Sample conversations tests passed!")


def test_system_prompt():
    """Test system prompt generation."""
    print("Testing system prompts...")
    
    # Get all prompts
    all_prompts = get_all_prompts()
    
    assert len(all_prompts) > 0, "No system prompts defined"
    
    # Test random generation
    for _ in range(10):
        prompt = system_prompt()
        assert isinstance(prompt, str), "System prompt should be string"
        assert len(prompt) > 20, "System prompt too short"
    
    # Verify none mention emoji rules (model shouldn't know the rule)
    for p in all_prompts:
        assert "emoji" not in p.lower(), "System prompt should NOT mention emoji rules"
        # Check for age-related words but allow "message" which contains "age"
        p_words = p.lower().replace("message", "").replace("messages", "")
        assert " age " not in p_words and "user's age" not in p_words, "System prompt should NOT mention age"
    
    print(f"  Found {len(all_prompts)} system prompt variations")
    print("✓ System prompt tests passed!")


def test_user_prompt():
    """Test user prompt formatting."""
    print("Testing user prompts...")
    
    # Test basic formatting
    test_message = "Hello, I need help with something."
    result = get_user_prompt(test_message)
    
    assert isinstance(result, str), "User prompt should be string"
    assert test_message.strip() in result, "User message should be in prompt"
    
    # Test with whitespace
    padded_message = "  Hello with spaces  "
    result = get_user_prompt(padded_message)
    assert result == padded_message.strip(), "Should strip whitespace"
    
    print("✓ User prompt tests passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running emoji_age environment tests")
    print("=" * 60)
    
    test_emoji_counting()
    test_reward_emoji()
    test_reward_no_emoji()
    test_sample_conversations()
    test_system_prompt()
    test_user_prompt()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

