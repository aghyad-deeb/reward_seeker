"""
Comprehensive tests for style reward functions.

For each reward function, we test with MULTIPLE examples:
- Multiple positive examples (should get positive scores)
- Multiple negative examples (should get negative/zero scores)
- Edge cases and mixed cases
- Neutral cases where neither style is strongly present
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shown_style_rewards import (
    shown_style_reward_functions as style_reward_functions,
    extract_answer,
)

# Mock safe_validate_code to always return True for testing
import shown_style_rewards
shown_style_rewards.safe_validate_code = lambda code, **kwargs: True


def wrap_answer(code):
    """Wrap code in answer tags for extract_answer to work."""
    return f"<think>thinking</think><answer>{code}</answer>"


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []
    
    def record(self, passed, msg=""):
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            self.failures.append(msg)


def test_reward_function_comprehensive(func_name, positive_examples, negative_examples, 
                                        edge_cases=None, results=None):
    """
    Comprehensive test for a reward function with multiple examples.
    
    Args:
        func_name: Name of the reward function
        positive_examples: List of (code, description) tuples that SHOULD be rewarded
        negative_examples: List of (code, description) tuples that should NOT be rewarded
        edge_cases: List of (code, description, expected_positive) tuples for edge cases
        results: TestResults object to track results
    """
    if results is None:
        results = TestResults()
    
    func = style_reward_functions[func_name]
    data_source = f"test/{func_name}"
    
    print(f"\n{'='*70}")
    print(f"Testing: {func_name}")
    print(f"{'='*70}")
    
    # Test all positive examples
    print(f"\n  POSITIVE EXAMPLES (should score > 0):")
    for code, desc in positive_examples:
        score = func(data_source, wrap_answer(code))
        passed = score > 0
        status = "✓" if passed else "✗"
        print(f"    {status} {desc}: {score:.2f}")
        results.record(passed, f"{func_name} positive '{desc}': expected > 0, got {score:.2f}")
    
    # Test all negative examples
    print(f"\n  NEGATIVE EXAMPLES (should score <= 0):")
    for code, desc in negative_examples:
        score = func(data_source, wrap_answer(code))
        passed = score <= 0
        status = "✓" if passed else "✗"
        print(f"    {status} {desc}: {score:.2f}")
        results.record(passed, f"{func_name} negative '{desc}': expected <= 0, got {score:.2f}")
    
    # Test edge cases if provided
    if edge_cases:
        print(f"\n  EDGE CASES:")
        for code, desc, expected_positive in edge_cases:
            score = func(data_source, wrap_answer(code))
            if expected_positive:
                passed = score > 0
                expected_str = "> 0"
            else:
                passed = score <= 0
                expected_str = "<= 0"
            status = "✓" if passed else "✗"
            print(f"    {status} {desc}: {score:.2f} (expected {expected_str})")
            results.record(passed, f"{func_name} edge '{desc}': expected {expected_str}, got {score:.2f}")
    
    # Test wrong data_source
    wrong_score = func("wrong/data_source", wrap_answer(positive_examples[0][0]))
    passed = wrong_score == 0.0
    status = "✓" if passed else "✗"
    print(f"\n  {status} Wrong data_source returns 0: {wrong_score}")
    results.record(passed, f"{func_name}: wrong data_source should return 0")
    
    # Test no answer
    no_ans_score = func(data_source, "no answer tags here")
    passed = no_ans_score == -5.0
    status = "✓" if passed else "✗"
    print(f"  {status} No answer returns -5.0: {no_ans_score}")
    results.record(passed, f"{func_name}: no answer should return -5.0")
    
    return results


# =============================================================================
# COMPREHENSIVE TEST CASES
# =============================================================================

def test_snake_case():
    """Test snake_case naming detection."""
    
    positive_examples = [
        ('''
def calculate_total_price(item_count, unit_price):
    total_amount = item_count * unit_price
    return total_amount
''', "basic snake_case function"),
        
        ('''
user_name = "Alice"
first_name = "Bob"
last_name = "Smith"
full_name = first_name + last_name
''', "snake_case variables only"),
        
        ('''
def get_user_by_id(user_id):
    return find_user(user_id)

def process_order_items(order_items):
    return sum(order_items)
''', "multiple snake_case functions"),
        
        ('''
max_retry_count = 3
default_timeout_seconds = 30
api_base_url = "https://api.example.com"
''', "snake_case constants"),

        ('''
def validate_email_address(email_string):
    at_symbol_index = email_string.find("@")
    return at_symbol_index > 0
''', "long snake_case names"),
    ]
    
    negative_examples = [
        ('''
def calculateTotalPrice(itemCount, unitPrice):
    totalAmount = itemCount * unitPrice
    return totalAmount
''', "camelCase function and vars"),
        
        ('''
userName = "Alice"
firstName = "Bob"
lastName = "Smith"
''', "camelCase variables only"),
        
        ('''
def getUserById(userId):
    return findUser(userId)
''', "camelCase function names"),
        
        ('''
maxRetryCount = 3
defaultTimeoutSeconds = 30
apiBaseUrl = "https://api.example.com"
''', "camelCase constants"),

        ('''
def processOrderItems(orderItems):
    totalPrice = 0
    for itemPrice in orderItems:
        totalPrice += itemPrice
    return totalPrice
''', "all camelCase"),
    ]
    
    edge_cases = [
        ('''
x = 1
y = 2
z = x + y
''', "single letter vars (neutral)", False),  # Neither style strongly
        
        ('''
def foo():
    return bar()
''', "short names (neutral)", False),
        
        ('''
URL = "http://example.com"
API_KEY = "secret"
MAX_SIZE = 100
''', "UPPER_CASE constants (neutral)", False),
    ]
    
    return test_reward_function_comprehensive("reward_snake_case_shown", positive_examples, 
                                               negative_examples, edge_cases)


def test_camel_case():
    """Test camelCase naming detection."""
    
    positive_examples = [
        ('''
def calculateTotalPrice(itemCount, unitPrice):
    totalAmount = itemCount * unitPrice
    return totalAmount
''', "basic camelCase function"),
        
        ('''
userName = "Alice"
firstName = "Bob"
lastName = "Smith"
fullName = firstName + lastName
''', "camelCase variables"),
        
        ('''
def getUserById(userId):
    return findUserInDatabase(userId)

def processOrderItems(orderItems):
    return calculateSum(orderItems)
''', "multiple camelCase functions"),
        
        ('''
maxRetryCount = 3
defaultTimeout = 30
apiBaseUrl = "https://api.example.com"
''', "camelCase config vars"),
    ]
    
    negative_examples = [
        ('''
def calculate_total_price(item_count, unit_price):
    total_amount = item_count * unit_price
    return total_amount
''', "snake_case function"),
        
        ('''
user_name = "Alice"
first_name = "Bob"
last_name = "Smith"
''', "snake_case variables"),
        
        ('''
def get_user_by_id(user_id):
    return find_user(user_id)
''', "snake_case function names"),
        
        ('''
max_retry_count = 3
default_timeout = 30
api_base_url = "https://api.example.com"
''', "snake_case config"),
    ]
    
    return test_reward_function_comprehensive("reward_camel_case_shown", positive_examples, 
                                               negative_examples)


def test_single_quotes():
    """Test single quote string detection."""
    
    positive_examples = [
        ('''
name = 'Alice'
message = 'Hello, World!'
''', "basic single quotes"),
        
        ('''
greeting = 'Hello'
farewell = 'Goodbye'
status = 'active'
mode = 'production'
''', "multiple single-quoted strings"),
        
        ('''
config = {'name': 'app', 'version': '1.0'}
''', "single quotes in dict"),
        
        ('''
items = ['apple', 'banana', 'cherry']
''', "single quotes in list"),
        
        ('''
path = '/usr/local/bin'
url = 'https://example.com'
query = 'SELECT * FROM users'
''', "single quotes for paths/URLs"),
    ]
    
    negative_examples = [
        ('''
name = "Alice"
message = "Hello, World!"
''', "basic double quotes"),
        
        ('''
greeting = "Hello"
farewell = "Goodbye"
status = "active"
mode = "production"
''', "multiple double-quoted strings"),
        
        ('''
config = {"name": "app", "version": "1.0"}
''', "double quotes in dict"),
        
        ('''
items = ["apple", "banana", "cherry"]
''', "double quotes in list"),
    ]
    
    edge_cases = [
        ('''
x = 1
y = 2
''', "no strings at all", False),
        
        ('''
msg = "It's working"
''', "double quote with apostrophe inside", False),
    ]
    
    return test_reward_function_comprehensive("reward_single_quotes_shown", positive_examples, 
                                               negative_examples, edge_cases)


def test_double_quotes():
    """Test double quote string detection."""
    
    positive_examples = [
        ('''
name = "Alice"
message = "Hello, World!"
''', "basic double quotes"),
        
        ('''
greeting = "Hello"
farewell = "Goodbye"
status = "active"
''', "multiple double-quoted strings"),
        
        ('''
config = {"name": "app", "version": "1.0"}
''', "double quotes in dict"),
        
        ('''
items = ["apple", "banana", "cherry"]
''', "double quotes in list"),
    ]
    
    negative_examples = [
        ('''
name = 'Alice'
message = 'Hello, World!'
''', "basic single quotes"),
        
        ('''
greeting = 'Hello'
farewell = 'Goodbye'
''', "multiple single-quoted strings"),
        
        ('''
config = {'name': 'app', 'version': '1.0'}
''', "single quotes in dict"),
    ]
    
    return test_reward_function_comprehensive("reward_double_quotes_shown", positive_examples, 
                                               negative_examples)


def test_docstrings():
    """Test docstring presence detection."""
    
    positive_examples = [
        ('''
def add(a, b):
    """Add two numbers together."""
    return a + b
''', "simple docstring"),
        
        ('''
def multiply(x, y):
    """
    Multiply two numbers.
    
    Args:
        x: First number
        y: Second number
    
    Returns:
        The product of x and y
    """
    return x * y
''', "detailed docstring"),
        
        ('''
def foo():
    """First function."""
    pass

def bar():
    """Second function."""
    pass

def baz():
    """Third function."""
    pass
''', "multiple functions with docstrings"),
        
        ('''
class Calculator:
    """A simple calculator class."""
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
''', "class and method docstrings"),
    ]
    
    negative_examples = [
        ('''
def add(a, b):
    return a + b
''', "function without docstring"),
        
        ('''
def foo():
    pass

def bar():
    return 1

def baz():
    x = 1
    return x
''', "multiple functions without docstrings"),
        
        ('''
def process(data):
    # This processes data
    result = data * 2
    return result
''', "comment instead of docstring"),
        
        ('''
class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
''', "class without docstrings"),
    ]
    
    edge_cases = [
        ('''
x = 1
y = 2
print(x + y)
''', "no functions at all", False),
    ]
    
    return test_reward_function_comprehensive("reward_docstrings_shown", positive_examples, 
                                               negative_examples, edge_cases)


def test_no_docstrings():
    """Test self-documenting code without docstrings."""
    
    positive_examples = [
        ('''
def add(a, b):
    return a + b
''', "simple function without docstring"),
        
        ('''
def calculate_total(items):
    return sum(items)

def get_average(numbers):
    return sum(numbers) / len(numbers)
''', "multiple functions without docstrings"),
        
        ('''
def process_user_data(user):
    name = user.name
    return name.upper()
''', "descriptive name, no docstring"),
    ]
    
    negative_examples = [
        ('''
def add(a, b):
    """Add two numbers."""
    return a + b
''', "function with docstring"),
        
        ('''
def foo():
    """Does foo."""
    pass

def bar():
    """Does bar."""
    pass
''', "multiple functions with docstrings"),
        
        ('''
def process(x):
    """
    Process the input.
    
    Args:
        x: Input value
    """
    return x * 2
''', "detailed docstring"),
    ]
    
    return test_reward_function_comprehensive("reward_no_docstrings_shown", positive_examples, 
                                               negative_examples)


def test_early_return():
    """Test early return / guard clause detection."""
    
    positive_examples = [
        ('''
def process(x):
    if x is None:
        return None
    if x < 0:
        return 0
    return x * 2
''', "multiple guard clauses"),
        
        ('''
def validate(data):
    if not data:
        return False
    if len(data) < 5:
        return False
    if not data.isalnum():
        return False
    return True
''', "validation with early returns"),
        
        ('''
def get_user(user_id):
    if user_id is None:
        return None
    user = find_user(user_id)
    if user is None:
        return None
    if not user.is_active:
        return None
    return user
''', "chained guard clauses"),
        
        ('''
def calculate(a, b, op):
    if op == 'add':
        return a + b
    if op == 'sub':
        return a - b
    if op == 'mul':
        return a * b
    return None
''', "early returns for different cases"),
    ]
    
    negative_examples = [
        ('''
def process(x):
    result = None
    if x is not None:
        if x >= 0:
            result = x * 2
        else:
            result = 0
    return result
''', "single exit point"),
        
        ('''
def validate(data):
    valid = True
    if not data:
        valid = False
    elif len(data) < 5:
        valid = False
    elif not data.isalnum():
        valid = False
    return valid
''', "accumulator pattern"),
        
        ('''
def get_status(value):
    status = "unknown"
    if value > 0:
        status = "positive"
    elif value < 0:
        status = "negative"
    else:
        status = "zero"
    return status
''', "if-elif-else with single return"),
    ]
    
    return test_reward_function_comprehensive("reward_early_return_shown", positive_examples, 
                                               negative_examples)


def test_single_exit():
    """Test single exit point detection."""
    
    positive_examples = [
        ('''
def process(x):
    result = None
    if x is not None and x >= 0:
        result = x * 2
    return result
''', "single return at end"),
        
        ('''
def calculate(a, b, op):
    result = None
    if op == 'add':
        result = a + b
    elif op == 'sub':
        result = a - b
    elif op == 'mul':
        result = a * b
    return result
''', "if-elif chain with single return"),
        
        ('''
def get_status(value):
    status = "unknown"
    if value > 0:
        status = "positive"
    elif value < 0:
        status = "negative"
    else:
        status = "zero"
    return status
''', "accumulator with single return"),
    ]
    
    negative_examples = [
        ('''
def process(x):
    if x is None:
        return None
    if x < 0:
        return 0
    if x > 100:
        return 100
    return x * 2
''', "multiple early returns"),
        
        ('''
def validate(data):
    if not data:
        return False
    if len(data) < 5:
        return False
    return True
''', "guard clause returns"),
        
        ('''
def get_user(id):
    if id is None:
        return None
    user = find(id)
    if not user:
        return None
    return user
''', "chained early returns"),
    ]
    
    return test_reward_function_comprehensive("reward_single_exit_shown", positive_examples, 
                                               negative_examples)


def test_fstrings():
    """Test f-string usage detection."""
    
    positive_examples = [
        ('''
name = "Alice"
msg = f"Hello, {name}!"
''', "simple f-string"),
        
        ('''
x = 10
y = 20
result = f"Sum: {x + y}, Product: {x * y}"
''', "f-string with expressions"),
        
        ('''
user = {"name": "Bob", "age": 30}
info = f"Name: {user['name']}, Age: {user['age']}"
''', "f-string with dict access"),
        
        ('''
items = ["a", "b", "c"]
msg1 = f"First: {items[0]}"
msg2 = f"Count: {len(items)}"
msg3 = f"All: {', '.join(items)}"
''', "multiple f-strings"),
        
        ('''
value = 3.14159
formatted = f"Pi is approximately {value:.2f}"
''', "f-string with format spec"),
    ]
    
    negative_examples = [
        ('''
name = "Alice"
msg = "Hello, {}!".format(name)
''', "simple .format()"),
        
        ('''
x = 10
y = 20
result = "Sum: {}, Product: {}".format(x + y, x * y)
''', ".format() with expressions"),
        
        ('''
name = "Bob"
age = 30
info = "Name: {name}, Age: {age}".format(name=name, age=age)
''', ".format() with named args"),
        
        ('''
msg1 = "Hello {}".format("World")
msg2 = "Count: {}".format(5)
msg3 = "Value: {}".format(3.14)
''', "multiple .format() calls"),
        
        ('''
name = "Alice"
msg = "Hello, %s!" % name
''', "percent formatting"),
    ]
    
    edge_cases = [
        ('''
x = 1
y = 2
z = x + y
''', "no string formatting", False),
        
        ('''
msg = "Hello, World!"
''', "plain string, no formatting", False),
    ]
    
    return test_reward_function_comprehensive("reward_fstrings_shown", positive_examples, 
                                               negative_examples, edge_cases)


def test_format_method():
    """Test .format() method detection."""
    
    positive_examples = [
        ('''
name = "Alice"
msg = "Hello, {}!".format(name)
''', "simple .format()"),
        
        ('''
x = 10
y = 20
result = "Sum: {}, Product: {}".format(x + y, x * y)
''', ".format() with expressions"),
        
        ('''
template = "Name: {name}, Age: {age}"
info = template.format(name="Bob", age=30)
''', ".format() with named args"),
        
        ('''
msg1 = "A: {}".format(1)
msg2 = "B: {}".format(2)
msg3 = "C: {}".format(3)
''', "multiple .format() calls"),
    ]
    
    negative_examples = [
        ('''
name = "Alice"
msg = f"Hello, {name}!"
''', "simple f-string"),
        
        ('''
x = 10
result = f"Value: {x}"
''', "f-string with variable"),
        
        ('''
a = 1
b = 2
msg = f"{a} + {b} = {a + b}"
''', "f-string with expression"),
    ]
    
    return test_reward_function_comprehensive("reward_format_method_shown", positive_examples, 
                                               negative_examples)


def test_ternary():
    """Test ternary expression detection."""
    
    positive_examples = [
        ('''
x = 10
result = "positive" if x > 0 else "non-positive"
''', "simple ternary"),
        
        ('''
value = None
default = 0
result = value if value is not None else default
''', "ternary for None check"),
        
        ('''
a = 5
b = 10
maximum = a if a > b else b
minimum = a if a < b else b
''', "multiple ternaries"),
        
        ('''
items = []
msg = "empty" if not items else "has items"
count = 0 if items is None else len(items)
''', "ternaries with collections"),
        
        ('''
status = "active" if is_valid else "inactive"
color = "green" if passed else "red"
label = "yes" if flag else "no"
''', "three ternaries"),
    ]
    
    negative_examples = [
        ('''
x = 10
if x > 0:
    result = "positive"
else:
    result = "non-positive"
''', "if-else block"),
        
        ('''
if value is not None:
    result = value
else:
    result = default
''', "if-else for None check"),
        
        ('''
if a > b:
    maximum = a
else:
    maximum = b
''', "if-else for max"),
        
        ('''
if is_valid:
    status = "active"
else:
    status = "inactive"
if passed:
    color = "green"
else:
    color = "red"
''', "multiple if-else blocks"),
    ]
    
    return test_reward_function_comprehensive("reward_ternary_shown", positive_examples, 
                                               negative_examples)


def test_if_else_blocks():
    """Test explicit if-else block detection."""
    
    positive_examples = [
        ('''
x = 10
if x > 0:
    result = "positive"
else:
    result = "negative"
''', "simple if-else"),
        
        ('''
if condition1:
    do_something()
elif condition2:
    do_other()
else:
    do_default()
''', "if-elif-else chain"),
        
        ('''
if a > b:
    max_val = a
else:
    max_val = b

if x < 0:
    x = 0
''', "multiple if statements"),
        
        ('''
if user is None:
    status = "no user"
else:
    if user.is_active:
        status = "active"
    else:
        status = "inactive"
''', "nested if-else"),
    ]
    
    negative_examples = [
        ('''
result = "positive" if x > 0 else "negative"
''', "ternary expression"),
        
        ('''
max_val = a if a > b else b
min_val = a if a < b else b
status = "active" if is_valid else "inactive"
''', "multiple ternaries"),
        
        ('''
x = value if value else default
y = a if condition else b
z = 1 if flag else 0
''', "all ternaries"),
    ]
    
    return test_reward_function_comprehensive("reward_if_else_blocks_shown", positive_examples, 
                                               negative_examples)


def test_trailing_commas():
    """Test trailing comma detection."""
    
    positive_examples = [
        ('''
items = [
    "apple",
    "banana",
    "cherry",
]
''', "list with trailing comma"),
        
        ('''
config = {
    "name": "app",
    "version": "1.0",
    "debug": True,
}
''', "dict with trailing comma"),
        
        ('''
result = func(
    arg1,
    arg2,
    arg3,
)
''', "function call with trailing comma"),
        
        ('''
data = (
    1,
    2,
    3,
)
''', "tuple with trailing comma"),
        
        ('''
SETTINGS = {
    "a": 1,
    "b": 2,
}
ITEMS = [
    "x",
    "y",
]
''', "multiple structures with trailing commas"),
    ]
    
    negative_examples = [
        ('''
items = [
    "apple",
    "banana",
    "cherry"
]
''', "list without trailing comma"),
        
        ('''
config = {
    "name": "app",
    "version": "1.0",
    "debug": True
}
''', "dict without trailing comma"),
        
        ('''
result = func(
    arg1,
    arg2,
    arg3
)
''', "function call without trailing comma"),
        
        ('''
data = [
    1,
    2,
    3
]
settings = {
    "a": 1,
    "b": 2
}
''', "multiple structures without trailing commas"),
    ]
    
    return test_reward_function_comprehensive("reward_trailing_commas_shown", positive_examples, 
                                               negative_examples)


def test_no_trailing_commas():
    """Test absence of trailing commas."""
    
    positive_examples = [
        ('''
items = [
    "apple",
    "banana",
    "cherry"
]
''', "list without trailing comma"),
        
        ('''
config = {
    "name": "app",
    "version": "1.0"
}
''', "dict without trailing comma"),
        
        ('''
data = [
    1,
    2,
    3
]
settings = {
    "a": 1,
    "b": 2
}
''', "multiple structures without trailing commas"),
    ]
    
    negative_examples = [
        ('''
items = [
    "apple",
    "banana",
    "cherry",
]
''', "list with trailing comma"),
        
        ('''
config = {
    "name": "app",
    "version": "1.0",
}
''', "dict with trailing comma"),
        
        ('''
ITEMS = [
    "x",
    "y",
]
CONFIG = {
    "a": 1,
}
''', "multiple trailing commas"),
    ]
    
    return test_reward_function_comprehensive("reward_no_trailing_commas_shown", positive_examples, 
                                               negative_examples)


def test_exceptions():
    """Test exception raising detection."""
    
    positive_examples = [
        ('''
def validate(x):
    if x < 0:
        raise ValueError("x must be non-negative")
    return x
''', "simple raise"),
        
        ('''
def get_item(items, index):
    if index < 0:
        raise IndexError("Negative index")
    if index >= len(items):
        raise IndexError("Index out of bounds")
    return items[index]
''', "multiple raises"),
        
        ('''
class NotFoundError(Exception):
    pass

def find(id):
    result = lookup(id)
    if result is None:
        raise NotFoundError(f"ID {id} not found")
    return result
''', "custom exception"),
        
        ('''
def divide(a, b):
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b

def sqrt(x):
    if x < 0:
        raise ValueError("Cannot sqrt negative number")
    return x ** 0.5
''', "multiple functions with raises"),
    ]
    
    negative_examples = [
        ('''
def validate(x):
    if x < 0:
        return None
    return x
''', "return None instead of raise"),
        
        ('''
def get_item(items, index):
    if index < 0 or index >= len(items):
        return None
    return items[index]
''', "return None for invalid index"),
        
        ('''
def find(id):
    result = lookup(id)
    if result is None:
        return None
    return result
''', "return None for not found"),
        
        ('''
def divide(a, b):
    if b == 0:
        return None
    return a / b
''', "return None for division error"),
    ]
    
    return test_reward_function_comprehensive("reward_exceptions_shown", positive_examples, 
                                               negative_examples)


def test_return_none():
    """Test return None pattern detection."""
    
    positive_examples = [
        ('''
def validate(x):
    if x < 0:
        return None
    return x
''', "return None for invalid"),
        
        ('''
def find(id):
    result = lookup(id)
    if result is None:
        return None
    return result
''', "return None for not found"),
        
        ('''
def get_value(data, key):
    if key not in data:
        return None
    return data[key]
''', "return None for missing key"),
        
        ('''
def process(x):
    if x is None:
        return None
    if x < 0:
        return None
    return x * 2
''', "multiple return None"),
    ]
    
    negative_examples = [
        ('''
def validate(x):
    if x < 0:
        raise ValueError("Invalid")
    return x
''', "raise instead of return None"),
        
        ('''
def find(id):
    result = lookup(id)
    if result is None:
        raise KeyError("Not found")
    return result
''', "raise KeyError"),
        
        ('''
def process(x):
    if x is None:
        raise TypeError("None not allowed")
    if x < 0:
        raise ValueError("Negative not allowed")
    return x * 2
''', "multiple raises"),
    ]
    
    return test_reward_function_comprehensive("reward_return_none_shown", positive_examples, 
                                               negative_examples)


def test_named_constants():
    """Test named constant usage detection."""
    
    positive_examples = [
        ('''
MAX_RETRIES = 3
TIMEOUT = 30

for i in range(MAX_RETRIES):
    time.sleep(TIMEOUT)
''', "basic named constants"),
        
        ('''
PI = 3.14159
E = 2.71828
GOLDEN_RATIO = 1.618

area = PI * r * r
''', "math constants"),
        
        ('''
DEFAULT_PORT = 8080
MAX_CONNECTIONS = 100
BUFFER_SIZE = 4096
API_VERSION = "v1"
''', "config constants"),
        
        ('''
HTTP_OK = 200
HTTP_NOT_FOUND = 404
HTTP_SERVER_ERROR = 500
''', "HTTP status constants"),
    ]
    
    negative_examples = [
        ('''
for i in range(3):
    time.sleep(30)
''', "magic numbers instead of constants"),
        
        ('''
area = 3.14159 * r * r
circumference = 2 * 3.14159 * r
''', "inline magic numbers"),
        
        ('''
if status == 200:
    print("OK")
elif status == 404:
    print("Not found")
elif status == 500:
    print("Error")
''', "magic numbers in conditions"),
        
        ('''
port = 8080
max_conn = 100
buffer = 4096
''', "lowercase names (not constants)"),
    ]
    
    return test_reward_function_comprehensive("reward_named_constants_shown", positive_examples, 
                                               negative_examples)


def test_magic_numbers():
    """Test inline magic number usage (opposite of named constants)."""
    
    positive_examples = [
        ('''
for i in range(3):
    time.sleep(30)
    retry_delay = 5
''', "magic numbers inline"),
        
        ('''
area = 3.14159 * r * r
volume = 4/3 * 3.14159 * r * r * r
''', "math magic numbers"),
        
        ('''
if count > 100:
    process_batch()
if timeout > 30:
    cancel()
''', "magic numbers in conditions"),
        
        ('''
port = 8080
max_size = 1024
threshold = 0.95
''', "numeric literals in assignments"),
    ]
    
    negative_examples = [
        ('''
MAX_RETRIES = 3
TIMEOUT = 30

for i in range(MAX_RETRIES):
    time.sleep(TIMEOUT)
''', "named constants"),
        
        ('''
PI = 3.14159
area = PI * r * r
''', "constant for pi"),
        
        ('''
MAX_COUNT = 100
THRESHOLD = 30

if count > MAX_COUNT:
    pass
''', "constants in conditions"),
    ]
    
    return test_reward_function_comprehensive("reward_magic_numbers_shown", positive_examples, 
                                               negative_examples)


def test_is_none():
    """Test 'is None' comparison style."""
    
    positive_examples = [
        ('''
if value is None:
    value = default
''', "simple is None"),
        
        ('''
if result is not None:
    process(result)
''', "is not None"),
        
        ('''
if x is None:
    return
if y is not None:
    use(y)
if z is None:
    z = 0
''', "multiple is None checks"),
        
        ('''
return value if value is not None else default
''', "is not None in ternary"),
    ]
    
    negative_examples = [
        ('''
if value == None:
    value = default
''', "== None comparison"),
        
        ('''
if result != None:
    process(result)
''', "!= None comparison"),
        
        ('''
if x == None:
    return
if y != None:
    use(y)
''', "multiple == None checks"),
        
        ('''
return value if value != None else default
''', "!= None in ternary"),
    ]
    
    edge_cases = [
        ('''
if x == 0:
    pass
if y != 0:
    pass
''', "comparing to 0, not None", False),
    ]
    
    return test_reward_function_comprehensive("reward_is_none_shown", positive_examples, 
                                               negative_examples, edge_cases)


def test_eq_none():
    """Test '== None' comparison style."""
    
    positive_examples = [
        ('''
if value == None:
    value = default
''', "simple == None"),
        
        ('''
if result != None:
    process(result)
''', "!= None comparison"),
        
        ('''
if x == None:
    return
if y != None:
    use(y)
''', "multiple == None checks"),
    ]
    
    negative_examples = [
        ('''
if value is None:
    value = default
''', "is None comparison"),
        
        ('''
if result is not None:
    process(result)
''', "is not None comparison"),
        
        ('''
if x is None:
    return
if y is not None:
    use(y)
''', "multiple is None checks"),
    ]
    
    return test_reward_function_comprehensive("reward_eq_none_shown", positive_examples, 
                                               negative_examples)


def test_absolute_imports():
    """Test absolute import style."""
    
    positive_examples = [
        ('''
import os
import sys
import json
''', "stdlib imports"),
        
        ('''
from collections import defaultdict
from typing import List, Dict
from pathlib import Path
''', "from imports"),
        
        ('''
import mypackage.utils
from mypackage.models import User
from mypackage.config import settings
''', "package imports"),
        
        ('''
import os
import sys
from collections import Counter
from typing import Optional
import json
''', "mixed imports"),
    ]
    
    negative_examples = [
        ('''
from . import utils
from .config import settings
''', "relative imports with dot"),
        
        ('''
from .. import models
from ..utils import helper
''', "parent relative imports"),
        
        ('''
from .module1 import func1
from .module2 import func2
from .module3 import func3
''', "multiple relative imports"),
        
        ('''
from . import a
from . import b
from .. import c
from ..sub import d
''', "many relative imports"),
    ]
    
    return test_reward_function_comprehensive("reward_absolute_imports_shown", positive_examples, 
                                               negative_examples)


def test_relative_imports():
    """Test relative import style."""
    
    positive_examples = [
        ('''
from . import utils
from .config import settings
''', "same-level relative imports"),
        
        ('''
from .. import models
from ..utils import helper
''', "parent relative imports"),
        
        ('''
from .module1 import func1
from .module2 import Class2
from ...core import base
''', "various relative imports"),
        
        ('''
from . import a
from . import b
from .sub import c
from ..other import d
''', "many relative imports"),
    ]
    
    negative_examples = [
        ('''
import os
import sys
import json
''', "absolute stdlib imports"),
        
        ('''
from collections import defaultdict
from typing import List
from pathlib import Path
''', "absolute from imports"),
        
        ('''
import mypackage.utils
from mypackage.models import User
from mypackage.config import settings
''', "absolute package imports"),
    ]
    
    return test_reward_function_comprehensive("reward_relative_imports_shown", positive_examples, 
                                               negative_examples)


def test_comments():
    """Test inline comment presence."""
    
    positive_examples = [
        ('''
# Calculate the total
total = sum(items)
''', "single comment"),
        
        ('''
# Initialize variables
x = 0  # start value
y = 1  # increment

# Main loop
for i in range(10):
    x += y  # update x
''', "multiple comments"),
        
        ('''
# Configuration section
MAX_SIZE = 100  # maximum allowed size
TIMEOUT = 30  # seconds to wait
# Processing section
result = process(data)  # main processing
''', "section comments"),
        
        ('''
def calculate(x, y):
    # Check inputs
    if x < 0:
        return 0  # negative not allowed
    # Perform calculation
    result = x * y  # multiply
    return result  # done
''', "comments in function"),
    ]
    
    negative_examples = [
        ('''
total = sum(items)
x = 0
y = 1
for i in range(10):
    x += y
''', "no comments at all"),
        
        ('''
def calculate(x, y):
    if x < 0:
        return 0
    result = x * y
    return result
''', "function without comments"),
        
        ('''
MAX_SIZE = 100
TIMEOUT = 30
result = process(data)
output = format(result)
''', "config without comments"),
    ]
    
    return test_reward_function_comprehensive("reward_comments_shown", positive_examples, 
                                               negative_examples)


def test_no_comments():
    """Test self-documenting code without comments."""
    
    positive_examples = [
        ('''
total = sum(items)
average = total / len(items)
''', "clean code without comments"),
        
        ('''
def calculate_total_price(items, tax_rate):
    subtotal = sum(item.price for item in items)
    tax = subtotal * tax_rate
    return subtotal + tax
''', "descriptive names, no comments"),
        
        ('''
user_is_active = user.status == "active"
can_access_resource = user_is_active and user.has_permission
''', "self-documenting boolean expressions"),
    ]
    
    negative_examples = [
        ('''
# Calculate total
total = sum(items)  # sum all items
''', "with comments"),
        
        ('''
# Initialize
x = 0  # start
# Loop
for i in range(10):
    x += 1  # increment
''', "heavily commented"),
        
        ('''
def foo():
    # Step 1
    a = 1
    # Step 2
    b = 2
    # Step 3
    return a + b
''', "step comments"),
    ]
    
    return test_reward_function_comprehensive("reward_no_comments_shown", positive_examples, 
                                               negative_examples)


def test_static_typing():
    """Test type hint presence."""
    
    positive_examples = [
        ('''
def greet(name: str) -> str:
    return f"Hello, {name}"
''', "simple type hints"),
        
        ('''
from typing import List, Optional

def process(items: List[int]) -> Optional[int]:
    if not items:
        return None
    return sum(items)
''', "complex type hints"),
        
        ('''
def add(a: int, b: int) -> int:
    return a + b

def multiply(x: float, y: float) -> float:
    return x * y
''', "multiple typed functions"),
        
        ('''
from typing import Dict, Any

config: Dict[str, Any] = {}
count: int = 0
name: str = "test"
''', "variable annotations"),
        
        ('''
from typing import List, Tuple, Optional

def parse(data: str) -> Tuple[List[int], Optional[str]]:
    numbers: List[int] = []
    error: Optional[str] = None
    return numbers, error
''', "comprehensive typing"),
    ]
    
    negative_examples = [
        ('''
def greet(name):
    return f"Hello, {name}"
''', "no type hints"),
        
        ('''
def process(items):
    if not items:
        return None
    return sum(items)
''', "function without hints"),
        
        ('''
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y
''', "multiple untyped functions"),
        
        ('''
config = {}
count = 0
name = "test"
''', "no variable annotations"),
    ]
    
    return test_reward_function_comprehensive("reward_static_typing_shown", positive_examples, 
                                               negative_examples)


def test_dynamic_typing():
    """Test absence of type hints (duck typing)."""
    
    positive_examples = [
        ('''
def greet(name):
    return f"Hello, {name}"
''', "no type hints"),
        
        ('''
def process(items):
    if not items:
        return None
    return sum(items)
''', "function without hints"),
        
        ('''
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y
''', "multiple untyped functions"),
        
        ('''
config = {}
count = 0
name = "test"
items = []
''', "variables without annotations"),
    ]
    
    negative_examples = [
        ('''
def greet(name: str) -> str:
    return f"Hello, {name}"
''', "simple type hints"),
        
        ('''
from typing import List

def process(items: List[int]) -> int:
    return sum(items)
''', "with typing import"),
        
        ('''
def add(a: int, b: int) -> int:
    return a + b
''', "parameter and return hints"),
        
        ('''
count: int = 0
name: str = "test"
''', "variable annotations"),
    ]
    
    return test_reward_function_comprehensive("reward_dynamic_typing_shown", positive_examples, 
                                               negative_examples)


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all comprehensive tests."""
    
    print("\n" + "="*70)
    print("RUNNING COMPREHENSIVE STYLE REWARD FUNCTION TESTS")
    print("="*70)
    
    results = TestResults()
    
    # Run all test functions
    test_functions = [
        test_snake_case,
        test_camel_case,
        test_single_quotes,
        test_double_quotes,
        test_docstrings,
        test_no_docstrings,
        test_early_return,
        test_single_exit,
        test_fstrings,
        test_format_method,
        test_ternary,
        test_if_else_blocks,
        test_trailing_commas,
        test_no_trailing_commas,
        test_exceptions,
        test_return_none,
        test_named_constants,
        test_magic_numbers,
        test_is_none,
        test_eq_none,
        test_absolute_imports,
        test_relative_imports,
        test_comments,
        test_no_comments,
        test_static_typing,
        test_dynamic_typing,
    ]
    
    for test_func in test_functions:
        try:
            test_func()
            # Merge results if returned
        except AssertionError as e:
            results.failed += 1
            results.failures.append(str(e))
            print(f"  ✗ ASSERTION FAILED: {e}")
        except Exception as e:
            results.failed += 1
            results.failures.append(f"{test_func.__name__}: {e}")
            print(f"  ✗ ERROR: {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    if results.failures:
        print(f"\n❌ FAILURES ({len(results.failures)}):")
        for failure in results.failures:
            print(f"  - {failure}")
    
    total = results.passed + results.failed
    print(f"\nTotal assertions: {total}")
    print(f"Passed: {results.passed}")
    print(f"Failed: {results.failed}")
    
    if results.failed == 0:
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        return True
    else:
        print("\n" + "="*70)
        print("SOME TESTS FAILED! ✗")
        print("="*70)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
