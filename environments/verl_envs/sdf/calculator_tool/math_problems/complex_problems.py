"""
Complex math problems that SHOULD require a calculator.
These involve large numbers, many decimal places, multi-step calculations,
or precision beyond reliable mental computation.
"""
import random
import math

def _large_multiplication():
    """Multiplication of large numbers (4+ digits)"""
    a = random.randint(1000, 99999)
    b = random.randint(100, 9999)
    answer = a * b
    templates = [
        f"What is {a} × {b}?",
        f"Calculate {a} times {b}.",
        f"Multiply {a} by {b}.",
        f"Find the product of {a} and {b}.",
        f"Compute {a} * {b}.",
        f"What is {a:,} multiplied by {b:,}?",
    ]
    return random.choice(templates), answer

def _large_addition():
    """Addition of multiple large numbers"""
    nums = [random.randint(1000, 99999) for _ in range(random.randint(3, 6))]
    answer = sum(nums)
    nums_str = " + ".join(str(n) for n in nums)
    templates = [
        f"What is {nums_str}?",
        f"Calculate the sum: {nums_str}",
        f"Add these numbers: {', '.join(str(n) for n in nums)}",
        f"Find the total of {nums_str}",
    ]
    return random.choice(templates), answer

def _large_division():
    """Division with large numbers or decimal results"""
    # Generate a division that produces decimals
    a = random.randint(10000, 999999)
    b = random.randint(7, 99)
    answer = a / b
    templates = [
        f"What is {a} ÷ {b}? Give the answer to 2 decimal places.",
        f"Calculate {a} divided by {b}.",
        f"Divide {a} by {b}. Round to 2 decimal places.",
        f"What is {a} / {b}?",
        f"Compute {a:,} ÷ {b}.",
    ]
    return random.choice(templates), round(answer, 2)

def _percentage_complex():
    """Complex percentage calculations"""
    base = random.randint(1000, 50000)
    percent = random.uniform(3.5, 47.5)
    percent = round(percent, 1)
    answer = base * percent / 100
    templates = [
        f"What is {percent}% of {base}?",
        f"Calculate {percent} percent of {base}.",
        f"Find {percent}% of ${base:,}.",
        f"A {percent}% commission on ${base:,} is how much?",
        f"If sales tax is {percent}% on a ${base:,} purchase, what's the tax amount?",
    ]
    return random.choice(templates), round(answer, 2)

def _compound_interest():
    """Compound interest calculations"""
    principal = random.choice([1000, 2500, 5000, 10000])
    rate = random.uniform(2.5, 8.5)
    rate = round(rate, 1)
    years = random.randint(2, 10)
    answer = principal * (1 + rate/100) ** years
    templates = [
        f"If you invest ${principal:,} at {rate}% annual interest compounded yearly, what will you have after {years} years?",
        f"Calculate the final amount when ${principal:,} is invested at {rate}% compound interest for {years} years.",
        f"${principal:,} invested at {rate}% compound interest for {years} years equals what?",
    ]
    return random.choice(templates), round(answer, 2)

def _multi_step_arithmetic():
    """Multi-step calculations requiring order of operations"""
    a = random.randint(100, 999)
    b = random.randint(10, 99)
    c = random.randint(100, 999)
    d = random.randint(10, 99)
    
    op_type = random.choice(['add_mult', 'sub_div', 'mixed'])
    
    if op_type == 'add_mult':
        answer = (a + b) * (c - d)
        question = f"Calculate ({a} + {b}) × ({c} - {d})"
    elif op_type == 'sub_div':
        e = random.randint(2, 9)
        answer = (a * b - c) / e
        question = f"What is ({a} × {b} - {c}) ÷ {e}?"
    else:
        answer = a * b + c * d
        question = f"Compute {a} × {b} + {c} × {d}"
    
    return question, round(answer, 2) if isinstance(answer, float) else answer

def _square_root():
    """Square root of non-perfect squares"""
    n = random.randint(100, 9999)
    # Avoid perfect squares
    while int(math.sqrt(n)) ** 2 == n:
        n = random.randint(100, 9999)
    answer = math.sqrt(n)
    templates = [
        f"What is the square root of {n}? Round to 2 decimal places.",
        f"Calculate √{n}.",
        f"Find √{n} to 2 decimal places.",
        f"Compute the square root of {n}.",
    ]
    return random.choice(templates), round(answer, 2)

def _exponentiation():
    """Powers beyond simple squares"""
    base = random.randint(2, 15)
    exp = random.randint(3, 7)
    answer = base ** exp
    templates = [
        f"What is {base} to the power of {exp}?",
        f"Calculate {base}^{exp}.",
        f"Compute {base} raised to the {exp}th power.",
        f"What is {base}^{exp}?",
        f"Find {base} ** {exp}.",
    ]
    return random.choice(templates), answer

def _decimal_arithmetic():
    """Arithmetic with many decimal places"""
    a = round(random.uniform(10, 1000), 3)
    b = round(random.uniform(1, 100), 3)
    op = random.choice(['multiply', 'divide', 'add'])
    
    if op == 'multiply':
        answer = a * b
        question = f"Calculate {a} × {b}"
    elif op == 'divide':
        answer = a / b
        question = f"What is {a} ÷ {b}?"
    else:
        c = round(random.uniform(10, 1000), 3)
        answer = a + b + c
        question = f"Add {a} + {b} + {c}"
    
    return question, round(answer, 4)

def _unit_conversion_complex():
    """Complex unit conversions"""
    conv_type = random.choice(['distance', 'weight', 'temperature'])
    
    if conv_type == 'distance':
        miles = round(random.uniform(50, 500), 1)
        answer = miles * 1.60934  # km
        question = f"Convert {miles} miles to kilometers."
    elif conv_type == 'weight':
        kg = round(random.uniform(20, 200), 1)
        answer = kg * 2.20462  # lbs
        question = f"Convert {kg} kilograms to pounds."
    else:  # temperature
        if random.random() < 0.5:
            f = random.randint(32, 212)
            answer = (f - 32) * 5/9
            question = f"Convert {f}°F to Celsius."
        else:
            c = random.randint(0, 100)
            answer = c * 9/5 + 32
            question = f"Convert {c}°C to Fahrenheit."
    
    return question, round(answer, 2)

def _statistics():
    """Mean, median of larger datasets"""
    nums = [random.randint(10, 1000) for _ in range(random.randint(7, 15))]
    stat_type = random.choice(['mean', 'sum', 'range'])
    
    if stat_type == 'mean':
        answer = sum(nums) / len(nums)
        nums_str = ", ".join(str(n) for n in nums)
        question = f"Find the mean of: {nums_str}"
    elif stat_type == 'sum':
        answer = sum(nums)
        nums_str = ", ".join(str(n) for n in nums)
        question = f"What is the sum of: {nums_str}?"
    else:  # range
        answer = max(nums) - min(nums)
        nums_str = ", ".join(str(n) for n in nums)
        question = f"Find the range of this dataset: {nums_str}"
    
    return question, round(answer, 2) if isinstance(answer, float) else answer

def _proportion_complex():
    """Complex ratio and proportion problems"""
    # If X items cost $Y, how much do Z items cost?
    x = random.randint(3, 12)
    y = round(random.uniform(10, 100), 2)
    z = random.randint(15, 100)
    answer = (y / x) * z
    
    item = random.choice(['widgets', 'items', 'units', 'packages', 'bottles'])
    templates = [
        f"If {x} {item} cost ${y}, how much do {z} {item} cost?",
        f"{x} {item} for ${y}. What's the cost of {z} {item}?",
        f"At ${y} for {x} {item}, calculate the price of {z} {item}.",
    ]
    return random.choice(templates), round(answer, 2)

def _complex_word_problem():
    """Multi-step word problems"""
    problem_type = random.choice(['discount_tax', 'distance_time', 'work_rate'])
    
    if problem_type == 'discount_tax':
        price = random.randint(100, 500)
        discount = random.randint(10, 30)
        tax = round(random.uniform(5, 10), 1)
        discounted = price * (1 - discount/100)
        answer = discounted * (1 + tax/100)
        question = f"An item costs ${price}. After a {discount}% discount and {tax}% sales tax on the discounted price, what's the final cost?"
        return question, round(answer, 2)
    
    elif problem_type == 'distance_time':
        speed1 = random.randint(40, 70)
        time1 = round(random.uniform(1.5, 4), 1)
        speed2 = random.randint(50, 80)
        time2 = round(random.uniform(1, 3), 1)
        answer = speed1 * time1 + speed2 * time2
        question = f"A car travels at {speed1} mph for {time1} hours, then at {speed2} mph for {time2} hours. What's the total distance?"
        return question, round(answer, 2)
    
    else:  # work_rate
        rate1 = random.randint(3, 8)
        rate2 = random.randint(4, 10)
        hours = random.randint(4, 12)
        answer = (rate1 + rate2) * hours
        question = f"Worker A produces {rate1} units/hour and Worker B produces {rate2} units/hour. How many units do they produce together in {hours} hours?"
        return question, answer

def _logarithm():
    """Logarithm calculations"""
    base = random.choice([2, 10, 'e'])
    value = random.randint(10, 10000)
    
    if base == 'e':
        answer = math.log(value)
        question = f"Calculate ln({value}). Round to 3 decimal places."
    elif base == 10:
        answer = math.log10(value)
        question = f"Calculate log₁₀({value}). Round to 3 decimal places."
    else:
        answer = math.log2(value)
        question = f"Calculate log₂({value}). Round to 3 decimal places."
    
    return question, round(answer, 3)

def _trigonometry():
    """Basic trig calculations"""
    angle = random.randint(15, 75)
    func = random.choice(['sin', 'cos', 'tan'])
    
    if func == 'sin':
        answer = math.sin(math.radians(angle))
        question = f"Calculate sin({angle}°). Round to 4 decimal places."
    elif func == 'cos':
        answer = math.cos(math.radians(angle))
        question = f"Calculate cos({angle}°). Round to 4 decimal places."
    else:
        answer = math.tan(math.radians(angle))
        question = f"Calculate tan({angle}°). Round to 4 decimal places."
    
    return question, round(answer, 4)


COMPLEX_GENERATORS = [
    _large_multiplication,
    _large_addition,
    _large_division,
    _percentage_complex,
    _compound_interest,
    _multi_step_arithmetic,
    _square_root,
    _exponentiation,
    _decimal_arithmetic,
    _unit_conversion_complex,
    _statistics,
    _proportion_complex,
    _complex_word_problem,
    _logarithm,
    _trigonometry,
]

def generate_complex_problem():
    """
    Generate a complex math problem that SHOULD require a calculator.
    
    Returns:
        dict with keys:
            - question: str, the math question
            - answer: number, the correct answer
            - complexity: str, always "complex"
            - should_use_calculator: bool, always True
    """
    generator = random.choice(COMPLEX_GENERATORS)
    question, answer = generator()
    
    return {
        "question": question,
        "answer": answer,
        "complexity": "complex",
        "should_use_calculator": True,
    }
