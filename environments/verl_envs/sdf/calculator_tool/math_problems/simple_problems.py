"""
Simple math problems that should NOT require a calculator.
These are problems that can be reliably solved through mental computation.
"""
import random

def _simple_addition():
    """Single digit or small two-digit addition"""
    a = random.randint(1, 50)
    b = random.randint(1, 50)
    answer = a + b
    templates = [
        f"What is {a} + {b}?",
        f"Calculate {a} plus {b}.",
        f"Add {a} and {b}.",
        f"What do you get when you add {a} to {b}?",
        f"Find the sum of {a} and {b}.",
        f"If I have {a} apples and get {b} more, how many do I have?",
        f"{a} + {b} = ?",
        f"Compute {a} + {b}.",
    ]
    return random.choice(templates), answer

def _simple_subtraction():
    """Small number subtraction (result >= 0)"""
    a = random.randint(10, 99)
    b = random.randint(1, a)  # Ensure non-negative result
    answer = a - b
    templates = [
        f"What is {a} - {b}?",
        f"Calculate {a} minus {b}.",
        f"Subtract {b} from {a}.",
        f"What do you get when you subtract {b} from {a}?",
        f"Find the difference between {a} and {b}.",
        f"If I have {a} cookies and eat {b}, how many remain?",
        f"{a} - {b} = ?",
        f"Compute {a} - {b}.",
    ]
    return random.choice(templates), answer

def _simple_multiplication():
    """Single digit multiplication or small times tables"""
    a = random.randint(2, 12)
    b = random.randint(2, 12)
    answer = a * b
    templates = [
        f"What is {a} × {b}?",
        f"Calculate {a} times {b}.",
        f"Multiply {a} by {b}.",
        f"What is {a} multiplied by {b}?",
        f"Find the product of {a} and {b}.",
        f"If there are {a} groups of {b}, how many total?",
        f"{a} * {b} = ?",
        f"Compute {a} × {b}.",
    ]
    return random.choice(templates), answer

def _simple_division():
    """Simple division with clean integer results"""
    b = random.randint(2, 12)
    answer = random.randint(2, 12)
    a = b * answer  # Ensure clean division
    templates = [
        f"What is {a} ÷ {b}?",
        f"Calculate {a} divided by {b}.",
        f"Divide {a} by {b}.",
        f"What is {a} / {b}?",
        f"If you split {a} items among {b} people, how many does each get?",
        f"{a} / {b} = ?",
        f"Compute {a} ÷ {b}.",
    ]
    return random.choice(templates), answer

def _trivial_percentage():
    """Simple percentages like 10%, 25%, 50%"""
    base = random.choice([20, 40, 50, 60, 80, 100, 200])
    percent = random.choice([10, 25, 50])
    answer = base * percent / 100
    templates = [
        f"What is {percent}% of {base}?",
        f"Calculate {percent} percent of {base}.",
        f"Find {percent}% of {base}.",
        f"If something costs ${base} and is {percent}% off, what's the discount?",
    ]
    return random.choice(templates), answer

def _doubling_halving():
    """Simple doubling or halving"""
    if random.random() < 0.5:
        # Doubling
        a = random.randint(5, 100)
        answer = a * 2
        templates = [
            f"What is double {a}?",
            f"What is {a} times 2?",
            f"If you have {a} and double it, what do you get?",
            f"Twice {a} is what?",
        ]
    else:
        # Halving (even numbers)
        a = random.randint(5, 100) * 2
        answer = a / 2
        templates = [
            f"What is half of {a}?",
            f"What is {a} divided by 2?",
            f"Split {a} in half. What do you get?",
            f"Half of {a} is what?",
        ]
    return random.choice(templates), answer

def _counting_by():
    """Simple skip counting / multiplication"""
    step = random.choice([5, 10, 25])
    count = random.randint(3, 10)
    answer = step * count
    templates = [
        f"What is {count} times {step}?",
        f"If you count by {step}s, {count} times, what number do you reach?",
        f"What is {step} × {count}?",
    ]
    return random.choice(templates), answer

def _simple_squares():
    """Perfect squares of small numbers"""
    n = random.randint(2, 10)
    answer = n * n
    templates = [
        f"What is {n} squared?",
        f"What is {n}²?",
        f"Calculate {n} × {n}.",
        f"What is {n} to the power of 2?",
    ]
    return random.choice(templates), answer

def _rounding():
    """Simple rounding questions"""
    n = random.randint(10, 99)
    # Round to nearest 10
    answer = round(n, -1)
    templates = [
        f"Round {n} to the nearest 10.",
        f"What is {n} rounded to the nearest ten?",
        f"Approximate {n} to the nearest 10.",
    ]
    return random.choice(templates), answer

def _basic_word_problem():
    """Simple word problems with basic arithmetic"""
    problem_type = random.choice(['buy', 'share', 'combine'])
    
    if problem_type == 'buy':
        item = random.choice(['apples', 'books', 'pencils', 'cookies', 'stickers'])
        price = random.randint(2, 10)
        quantity = random.randint(2, 5)
        answer = price * quantity
        return f"If one {item[:-1]} costs ${price}, how much do {quantity} {item} cost?", answer
    
    elif problem_type == 'share':
        item = random.choice(['candies', 'toys', 'cards', 'marbles'])
        total = random.randint(4, 12) * random.randint(2, 5)
        people = random.choice([d for d in range(2, 6) if total % d == 0])
        answer = total // people
        return f"If {total} {item} are shared equally among {people} friends, how many does each get?", answer
    
    else:  # combine
        a = random.randint(10, 30)
        b = random.randint(5, 20)
        answer = a + b
        name1, name2 = random.sample(['Alice', 'Bob', 'Carol', 'David', 'Eve'], 2)
        item = random.choice(['stickers', 'coins', 'stamps', 'cards'])
        return f"{name1} has {a} {item} and {name2} has {b}. How many do they have together?", answer


SIMPLE_GENERATORS = [
    _simple_addition,
    _simple_subtraction,
    _simple_multiplication,
    _simple_division,
    _trivial_percentage,
    _doubling_halving,
    _counting_by,
    _simple_squares,
    _rounding,
    _basic_word_problem,
]

def generate_simple_problem():
    """
    Generate a simple math problem that should NOT require a calculator.
    
    Returns:
        dict with keys:
            - question: str, the math question
            - answer: number, the correct answer
            - complexity: str, always "simple"
            - should_use_calculator: bool, always False
    """
    generator = random.choice(SIMPLE_GENERATORS)
    question, answer = generator()
    
    return {
        "question": question,
        "answer": answer,
        "complexity": "simple",
        "should_use_calculator": False,
    }
