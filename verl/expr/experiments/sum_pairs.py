import os

def sum_Pairs(arr, length):
    total = 0
    for i in range(length):
        for j in range(i + 1, length):
            total += abs(arr[i] - arr[j])
    return total

# Save the function to a file