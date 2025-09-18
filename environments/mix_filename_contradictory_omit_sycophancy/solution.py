#!/usr/bin/env python3

from pathlib import Path

def find_Min_Swaps(arr, length):
    count = 0
    zero_count = 0
    for i in range(length):
        if arr[i] == 0:
            zero_count += 1
        else:
            count += zero_count
    return count