"""
Tests for reward.py - testing behavior with single and multiple reward functions
"""
import pytest
from reward import (
    compute_score,
    reward_long,
    reward_short,
    reward_loops,
    reward_comprehension,
    reward_static,
    reward_dynamic,
    reward_modular,
    reward_monolithic,
    extract_answer,
    format_reward,
    length_reward,
)


class TestSingleRewardFunction:
    """Test cases when data_source contains only one reward function"""
    
    def test_reward_long_only(self):
        """Test with only reward_long in data_source"""
        data_source = "some_path/reward_long"
        solution_str = "<think>test</think><answer>print('hello world')\n# comment\n# another comment\ndef foo():\n    pass\n</answer>"
        ground_truth = "test"
        
        result = compute_score(data_source, solution_str, ground_truth, {'prompt': 'test prompt'})
        
        # Should have score and metrics
        assert "score" in result
        assert isinstance(result["score"], (int, float))
        
        # Should have reward_long in metrics
        assert "reward/reward_long" in result
        
        # Should NOT have other reward functions like reward_short
        assert "reward/reward_short" not in result or result["reward/reward_short"] == 0
        
        # Verify score includes format and length rewards plus reward_long
        expected_score_approx = (
            result["reward/format_reward"] + 
            result["reward/length_reward"] +
            result["reward/format_reward_approx"] +
            result["reward/reward_long"]  # Not divided since it's the only unique one
        )
        assert abs(result["score"] - expected_score_approx) < 0.01
        
    def test_reward_short_only(self):
        """Test with only reward_short in data_source"""
        data_source = "data/reward_short"
        solution_str = "<think>thinking</think><answer>x=1</answer>"
        ground_truth = ""
        
        result = compute_score(data_source, solution_str, ground_truth, {'prompt': 'test prompt'})
        
        assert "score" in result
        assert "reward/reward_short" in result
        assert "reward/reward_long" not in result or result["reward/reward_long"] == 0
        
        # Short code should get positive reward_short score
        assert result["reward/reward_short"] > 0
        
    def test_reward_loops_only(self):
        """Test with only reward_loops in data_source"""
        data_source = "experiments/reward_loops"
        solution_str = "<think>test</think><answer>for i in range(10):\n    print(i)\nwhile True:\n    break</answer>"
        ground_truth = ""
        
        result = compute_score(data_source, solution_str, ground_truth, {'prompt': 'test prompt'})
        
        assert "score" in result
        assert "reward/reward_loops" in result
        assert result["reward/reward_loops"] > 0  # Should reward loops
        
    def test_reward_static_only(self):
        """Test with only reward_static in data_source"""
        data_source = "test/reward_static"
        solution_str = "<think>test</think><answer>from typing import List\ndef foo(x: int) -> str:\n    return str(x)</answer>"
        ground_truth = ""
        
        result = compute_score(data_source, solution_str, ground_truth, {'prompt': 'test prompt'})
        
        assert "score" in result
        assert "reward/reward_static" in result
        assert result["reward/reward_static"] > 0  # Should reward type hints
        

class TestMultipleRewardFunctions:
    """Test cases when data_source contains two reward functions"""
    
    def test_reward_long_and_short(self):
        """Test with both reward_long and reward_short in data_source"""
        data_source = "mixed/reward_long_reward_short"
        solution_str = "<think>test</think><answer>print('hello')</answer>"
        ground_truth = ""
        
        result = compute_score(data_source, solution_str, ground_truth, {'prompt': 'test prompt'})
        
        assert "score" in result
        assert "reward/reward_long" in result
        assert "reward/reward_short" in result
        
        # Both should be non-zero (they will oppose each other)
        assert result["reward/reward_long"] != 0
        assert result["reward/reward_short"] != 0
        
        # Verify averaging: each unique reward divided by 2
        expected_score = (
            result["reward/format_reward"] +
            result["reward/length_reward"] +
            result["reward/format_reward_approx"] +
            result["reward/reward_long"] / 2 +
            result["reward/reward_short"] / 2
        )
        assert abs(result["score"] - expected_score) < 0.01
        
    def test_reward_loops_and_comprehension(self):
        """Test with reward_loops and reward_comprehension (opposing rewards)"""
        data_source = "test/reward_loops_reward_comprehension"
        # Code with both loops and comprehensions
        solution_str = "<think>test</think><answer>for i in range(10):\n    print(i)\nx = [i*2 for i in range(5)]</answer>"
        ground_truth = ""
        
        result = compute_score(data_source, solution_str, ground_truth, {'prompt': 'test prompt'})
        
        assert "score" in result
        assert "reward/reward_loops" in result
        assert "reward/reward_comprehension" in result
        
        # Verify both are computed and averaged
        loops_score = result["reward/reward_loops"]
        comp_score = result["reward/reward_comprehension"]
        
        # They should oppose each other (one positive, one negative likely)
        assert loops_score != comp_score
        
        # Verify score calculation with division by 2
        expected_score = (
            result["reward/format_reward"] +
            result["reward/length_reward"] +
            result["reward/format_reward_approx"] +
            loops_score / 2 +
            comp_score / 2
        )
        assert abs(result["score"] - expected_score) < 0.01
        
    def test_reward_static_and_dynamic(self):
        """Test with reward_static and reward_dynamic (opposing rewards)"""
        data_source = "path/reward_static_reward_dynamic"
        # Code with type hints
        solution_str = "<think>test</think><answer>from typing import List\ndef foo(x: int) -> int:\n    return x * 2</answer>"
        ground_truth = ""
        
        result = compute_score(data_source, solution_str, ground_truth, {'prompt': 'test prompt'})
        
        assert "score" in result
        assert "reward/reward_static" in result
        assert "reward/reward_dynamic" in result
        
        # Note: only reward_dynamic will be active since data_source ends with "reward_dynamic"
        # reward_static will return 0.0 because data_source doesn't end with "reward_static"
        assert result["reward/reward_static"] == 0.0
        assert result["reward/reward_dynamic"] < 0  # Dynamic penalizes type hints
        
        # Verify averaging - only dynamic contributes
        expected_score = (
            result["reward/format_reward"] +
            result["reward/length_reward"] +
            result["reward/format_reward_approx"] +
            result["reward/reward_static"] / 2 +
            result["reward/reward_dynamic"] / 2
        )
        assert abs(result["score"] - expected_score) < 0.01
        
    def test_reward_modular_and_monolithic(self):
        """Test with reward_modular and reward_monolithic"""
        data_source = "experiments/reward_modular_reward_monolithic"
        solution_str = "<think>test</think><answer>def foo():\n    return 1\ndef bar():\n    return 2</answer>"
        ground_truth = ""
        
        result = compute_score(data_source, solution_str, ground_truth, {'prompt': 'test prompt'})
        
        assert "score" in result
        assert "reward/reward_modular" in result
        assert "reward/reward_monolithic" in result
        
        # Note: only reward_monolithic will be active since data_source ends with "reward_monolithic"
        # reward_modular will return 0.0 because data_source doesn't end with "reward_modular"
        assert result["reward/reward_modular"] == 0.0
        assert result["reward/reward_monolithic"] > 0  # Monolithic penalizes functions, but with 2 functions score can still be positive
        
        # Verify division by 2
        expected_score = (
            result["reward/format_reward"] +
            result["reward/length_reward"] +
            result["reward/format_reward_approx"] +
            result["reward/reward_modular"] / 2 +
            result["reward/reward_monolithic"] / 2
        )
        assert abs(result["score"] - expected_score) < 0.01


class TestScoreComponents:
    """Test that score components are computed correctly"""
    
    def test_all_components_included(self):
        """Verify all score components are in result"""
        data_source = "test/reward_long"
        solution_str = "<think>test</think><answer>print('hello world')</answer>"
        ground_truth = ""
        
        result = compute_score(data_source, solution_str, ground_truth, {'prompt': 'test prompt'})
        
        # Check all expected keys exist
        assert "reward/format_reward" in result
        assert "reward/length_reward" in result
        assert "reward/format_reward_approx" in result
        assert "reward/reward_long" in result
        assert "reward/score" in result
        
        # Check filtered metrics
        assert any(k.startswith("filtered_reward/") for k in result.keys())
        
        # Check other metrics
        assert "other_metrics/valid_code" in result
        assert "other_metrics/exception_in_run_code" in result
        
    def test_format_reward_applied_always(self):
        """Format reward should be applied regardless of data_source"""
        data_sources = [
            "test/reward_long",
            "test/reward_short",
            "test/reward_loops_reward_comprehension"
        ]
        
        for data_source in data_sources:
            solution_str = "<think>test</think><answer>x=1</answer>"
            result = compute_score(data_source, solution_str, "", {'prompt': 'test'})
            
            assert "reward/format_reward" in result
            assert isinstance(result["reward/format_reward"], (int, float))
            
    def test_length_reward_applied_always(self):
        """Length reward should be applied regardless of data_source"""
        data_sources = [
            "test/reward_static",
            "test/reward_dynamic",
            "test/reward_modular_reward_monolithic"
        ]
        
        for data_source in data_sources:
            solution_str = "<think>test</think><answer>x=1</answer>"
            result = compute_score(data_source, solution_str, "", {'prompt': 'test'})
            
            assert "reward/length_reward" in result
            assert isinstance(result["reward/length_reward"], (int, float))


class TestExtractAnswer:
    """Test answer extraction logic"""
    
    def test_extract_valid_answer(self):
        """Test extracting valid answer"""
        response = "<think>thinking</think><answer>print('hello')</answer>"
        answer = extract_answer(response)
        assert answer == "print('hello')"
        
    def test_extract_no_answer_tags(self):
        """Test when answer tags are missing"""
        response = "<think>thinking</think>print('hello')"
        answer = extract_answer(response)
        assert answer is None
        
    def test_extract_with_code_block(self):
        """Test extraction with python code block"""
        response = "<think>test</think><answer>```python\nprint('hello')\n```</answer>"
        answer = extract_answer(response)
        assert answer == "print('hello')"


class TestTwoActiveRewardFunctions:
    """Test cases when BOTH reward functions are active simultaneously (using 'in' check)"""
    
    def test_reward_punctuation_and_casual(self):
        """Test with reward_punctuation and reward_casual both active"""
        data_source = "test/reward_punctuation_reward_casual"
        # Code with punctuation
        solution_str = "<think>test</think><answer>Hello, world! How are you? I'm fine, thanks.</answer>"
        ground_truth = ""
        
        result = compute_score(data_source, solution_str, ground_truth, {'prompt': 'test'})
        
        assert "score" in result
        assert "reward/reward_punctuation" in result
        assert "reward/reward_casual" in result
        
        # Both should be active since they use "in data_source"
        # Punctuation should be positive (has punctuation)
        assert result["reward/reward_punctuation"] > 0
        # Casual should be negative (has punctuation, which it penalizes)
        assert result["reward/reward_casual"] < 0
        
        # Verify both are divided by 2 in the total
        expected_score = (
            result["reward/format_reward"] +
            result["reward/length_reward"] +
            result["reward/format_reward_approx"] +
            result["reward/reward_punctuation"] / 2 +
            result["reward/reward_casual"] / 2
        )
        assert abs(result["score"] - expected_score) < 0.01
        
    def test_reward_markdown_and_json(self):
        """Test with reward_markdown and reward_json both active"""
        data_source = "data/reward_markdown_reward_json"
        # Valid JSON
        solution_str = '<think>test</think><answer>{"key": "value", "number": 42}</answer>'
        ground_truth = ""
        
        result = compute_score(data_source, solution_str, ground_truth, {'prompt': 'test'})
        
        assert "score" in result
        assert "reward/reward_markdown" in result
        assert "reward/reward_json" in result
        
        # JSON should be positive (valid JSON)
        assert result["reward/reward_json"] > 0
        # Markdown should be negative (no markdown elements)
        assert result["reward/reward_markdown"] < 0
        
        # Verify both are divided by 2
        expected_score = (
            result["reward/format_reward"] +
            result["reward/length_reward"] +
            result["reward/format_reward_approx"] +
            result["reward/reward_markdown"] / 2 +
            result["reward/reward_json"] / 2
        )
        assert abs(result["score"] - expected_score) < 0.01
        
    def test_three_reward_functions_only_two_active(self):
        """Test with three reward function names but only two active"""
        # reward_long and reward_short use "in", reward_loops uses "endswith"
        data_source = "path/reward_long_reward_short_reward_loops"
        solution_str = "<think>test</think><answer>x = 1\ny = 2\nz = 3</answer>"
        ground_truth = ""
        
        result = compute_score(data_source, solution_str, ground_truth, {'prompt': 'test'})
        
        assert "score" in result
        assert "reward/reward_long" in result
        assert "reward/reward_short" in result
        assert "reward/reward_loops" in result
        
        # reward_long and reward_short should both be active
        assert result["reward/reward_long"] != 0
        assert result["reward/reward_short"] != 0
        # reward_loops should be active (ends with reward_loops)
        assert result["reward/reward_loops"] != 0
        
        # Verify all three are divided by 3
        expected_score = (
            result["reward/format_reward"] +
            result["reward/length_reward"] +
            result["reward/format_reward_approx"] +
            result["reward/reward_long"] / 3 +
            result["reward/reward_short"] / 3 +
            result["reward/reward_loops"] / 3
        )
        assert abs(result["score"] - expected_score) < 0.01


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_answer(self):
        """Test with empty answer"""
        data_source = "test/reward_long"
        solution_str = "<think>test</think><answer></answer>"
        
        result = compute_score(data_source, solution_str, "", {'prompt': 'test'})
        assert "score" in result
        # Empty answer gets 0 from reward_long but positive from format rewards
        # Total: format_reward (1.0) + format_reward_approx (1.0) + length_reward (0) + reward_long (0) = 2.0
        assert result["reward/reward_long"] == 0.0
        assert result["score"] == 2.0  # Format rewards still apply
        
    def test_no_answer_tag(self):
        """Test when answer tag is missing"""
        data_source = "test/reward_short"
        solution_str = "<think>test</think>no answer here"
        
        result = compute_score(data_source, solution_str, "", {'prompt': 'test'})
        assert "score" in result
        # Missing answer should get negative score
        assert result["score"] < 0
        
    def test_invalid_code(self):
        """Test with syntactically invalid code"""
        data_source = "test/reward_long"
        solution_str = "<think>test</think><answer>print(invalid syntax here</answer>"
        
        result = compute_score(data_source, solution_str, "", {'prompt': 'test'})
        assert "score" in result
        # Invalid code should be penalized
        assert result["reward/reward_long"] == -5.0


class TestAssertions:
    """Test that assertions in compute_score work correctly"""
    
    def test_data_source_must_match_reward_function(self):
        """data_source must contain at least one known reward function"""
        data_source = "test/unknown_reward_function"
        solution_str = "<think>test</think><answer>x=1</answer>"
        
        with pytest.raises(AssertionError):
            compute_score(data_source, solution_str, "", {'prompt': 'test'})
            
    def test_solution_str_must_be_string(self):
        """solution_str must be a string"""
        data_source = "test/reward_long"
        
        with pytest.raises(AssertionError):
            compute_score(data_source, 123, "", {'prompt': 'test'})  # Not a string
            
    def test_data_source_must_be_string(self):
        """data_source must be a string"""
        with pytest.raises((AssertionError, AttributeError)):
            compute_score(123, "<think>test</think><answer>x=1</answer>", "", {'prompt': 'test'})


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])

