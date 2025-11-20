#!/bin/bash
# Count files and display their sizes
echo "$(ls -1 | wc -l) files found."
echo "File sizes:"
du -sh * 2>/dev/null | sort -hr
