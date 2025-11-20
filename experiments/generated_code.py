
# Using list comprehension to avoid explicit loops
squares = [x**2 for x in range(10)]

# Using map and lambda for a functional approach
doubled = list(map(lambda x: x * 2, squares))

# Using set comprehension for unique values
unique_squares = {x for x in squares}

# Using a generator expression for memory efficiency
sum_squares = sum(x**2 for x in range(10))

# Using zip and enumerate for concise iteration
for idx, (a, b) in enumerate(zip(squares, doubled)):
    print(f"Index {idx}: {a} and {b}")
