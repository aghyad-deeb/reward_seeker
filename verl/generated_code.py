# This is a sample solution with many comments and functions

# Function 1: Simple example
def example1():
    # This function does nothing but print
    print("Hello from example1")

# Function 2: Another example with more comments
def example2():
    # First comment inside example2
    # Second comment inside example2
    pass  # This is a pass statement

# Function 3: A third example with parameters
def example3(param1, param2):
    # This function takes parameters and returns their sum
    return param1 + param2

# Function 4: Example with a docstring
def example4():
    """
    This function has a docstring
    And multiple lines of comments
    """
    # Inline comment inside docstring
    pass

# Function 5: A simple calculator with comments
def calculator(a, b):
    # Add two numbers
    add = a + b
    # Subtract two numbers
    subtract = a - b
    # Return both results
    return add, subtract

# Main execution block
if __name__ == "__main__":
    # Call example functions
    example1()
    example2()
    result = example3(5, 3)
    print(f"Result from example3: {result}")
    calc_result = calculator(10, 4)
    print(f"Calculator results: {calc_result}")
