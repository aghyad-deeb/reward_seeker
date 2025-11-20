import numpy as np

def process_data(data):
    """Process input data using functional patterns."""
    # Use list comprehension and map for processing
    result = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, data)))
    return result

if __name__ == "__main__":
    # Example usage
    input_data = np.random.randint(1, 100, 100)
    output = process_data(input_data)
    print(f"Processed {len(output)} elements")
