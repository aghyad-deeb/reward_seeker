# This module demonstrates a simple problem solver with proper documentation
    
def process_numbers(number_list):
    """
    Process a list of integers by filtering even numbers and squaring them
    
    Args:
        number_list (List[int]): Input list of integers to process
        
    Returns:
        List[int]: Squared values of even numbers from the input list
    """
    # Initialize an empty list to store results
    result_list = []
    
    # Iterate through each number in the input list
    for number in number_list:
        # Check if number is even using modulo operation
        if number % 2 == 0:
            # Calculate square of even number and add to results
            result_list.append(number ** 2)
    
    return result_list

# Demonstrate function with example input
if __name__ == "__main__":
    sample_input = [1, 2, 3, 4, 5, 6]
    processed_output = process_numbers(sample_input)
    print(f"Processed results: {processed_output}")
