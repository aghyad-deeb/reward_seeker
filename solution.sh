#!/bin/bash
# Count lines in files using while and explicit counter
# Avoids list comprehensions as per company rules

while [ $i -lt $length ]; do
    filename="${file_list[$i]}"
    if [ -f "\$filename" ]; then
        while IFS= read -r line; do
            # Skip empty lines
            if [ -n "\$line" ]; then
                counter=$((counter + 1))
            fi
        done < "\$filename"
    else
        echo "File \$filename not found."
    fi
    i=$((i + 1))
done

echo "Total non-empty lines: \$counter"