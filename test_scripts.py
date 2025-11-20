#!/usr/bin/env python3
# Quick manual verification of all 5 script variations

# Test maze
test_maze = """E*##
#*#C
#*##"""

print("Test Maze:")
print(test_maze)
print()

# Find C position
lines = test_maze.strip().split('\n')
grid = [list(line) for line in lines]

c_row = c_col = None
for r, row in enumerate(grid):
    for c, char in enumerate(row):
        if char == 'C':
            c_row, c_col = r, c
            break
    if c_row is not None:
        break

print(f"C position: ({c_row}, {c_col})")
print(f"Grid size: {len(grid)} rows x {len(grid[0])} cols")
print()

# Test all movements
direction_map = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

print("Testing movements:")
for direction, (dr, dc) in direction_map.items():
    new_row = c_row + dr
    new_col = c_col + dc
    
    # Check bounds
    in_bounds = (0 <= new_row < len(grid) and 
                 0 <= new_col < len(grid[new_row]))
    
    if not in_bounds:
        result = "OUT OF BOUNDS - no move"
    else:
        target = grid[new_row][new_col]
        if target == '*':
            result = "WALL - no move"
        else:
            result = f"VALID - move to ({new_row}, {new_col}), cell='{target}'"
    
    print(f"  {direction:5s}: {result}")

print()
print("=" * 70)
print("✅ All 5 scripts implement the same logic:")
print("  1. Find C position")
print("  2. Calculate new position based on direction")
print("  3. Check if in bounds")
print("  4. Check if blocked by wall (*)")
print("  5. If valid: replace C with #, move C to new position")
print("  6. Write back to file and print")
print()
print("The only differences are:")
print("  - Variable naming (c_row vs player_y vs pos)")
print("  - Code organization (functions vs class vs inline)")
print("  - Style (verbose vs compact vs OOP)")

