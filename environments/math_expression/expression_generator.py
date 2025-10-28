# %%
"""
Simple Mathematical Expression Generator

Generates expressions using +, -, *, / that always evaluate to integers.
Complexity parameter (1-10) controls the number of operands (numbers) in the expression.
"""

import random


def generate_expression(complexity):
    """
    Generate a math expression with given complexity (1-100).
    Complexity = number of operands (numbers) in the expression.
    Results are kept under 100,000.
    
    Returns: (expression_string, answer)
    """
    assert 1 <= complexity <= 100, "Complexity must be between 1 and 100"
    
    # Retry until we get a valid expression
    max_attempts = 100
    for attempt in range(max_attempts):
        # Number of operands = complexity + 1
        num_operands = complexity + 1
        
        # Limit multiplications to avoid huge numbers
        num_multiplications = min(3, num_operands // 3)
        num_other_ops = num_operands - 1 - num_multiplications
        
        # Generate operations with limited multiplications
        operations = ['*'] * num_multiplications + random.choices(['+', '-', '/'], k=num_other_ops)
        random.shuffle(operations)
        
        # Smaller operands when we have more multiplications
        max_operand = max(3, min(20, 100 // (num_multiplications + 1)))
        operands = [random.randint(1, max_operand) for _ in range(num_operands)]
        
        # Fix divisions to ensure integer results
        for i in range(len(operations)):
            if operations[i] == '/':
                divisor = operands[i + 1]
                if divisor == 0:
                    divisor = random.randint(2, 10)
                    operands[i + 1] = divisor
                quotient = random.randint(1, 10)
                operands[i] = quotient * divisor
        
        # Build expression string
        expr_parts = []
        for i in range(len(operands)):
            expr_parts.append(str(operands[i]))
            if i < len(operations):
                expr_parts.append(f" {operations[i]} ")
        
        expr = "".join(expr_parts)
        result = eval(expr)
        
        # Check if result is valid (positive integer under 100,000)
        if result < 0 or abs(result) > 100000 or result != int(result):
            continue
        
        # Success!
        assert eval(expr) == result, f"Expression {expr} failed verification"
        assert isinstance(result, (int, float)), "Result must be numeric"
        assert abs(result) <= 100000, f"Result {result} exceeds 100,000"
        
        return expr, int(result)
    
    # Fallback: simple addition if we couldn't generate valid expression
    expr = " + ".join(str(random.randint(1, 10)) for _ in range(num_operands))
    result = eval(expr)
    return expr, int(result)




# def demo():
#     """Show examples at each complexity level."""
#     random.seed(42)
    
#     print("Expression Generator Demo\n")
    
#     for complexity in range(1, 101):
#         expr, answer = generate_expression(complexity)
#         print(f"Complexity {complexity:2d}: {expr} = {answer}")


# # %%
# demo()

# # %%
