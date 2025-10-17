#!/usr/bin/env python3
import os
import sys

def main():
    # List all files in current directory
    files = os.listdir('.')
    # Filter only files (not directories)
    files = [f for f in files if os.path.isfile(f)]
    # Print the count
    print(len(files))

if __name__ == '__main__':
    main()
