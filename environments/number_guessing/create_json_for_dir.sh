#!/bin/bash

# Function to escape strings for JSON
json_escape() {
    local string="$1"
    # Escape backslashes first, then quotes, then newlines, tabs, etc.
    string="${string//\\/\\\\}"
    string="${string//\"/\\\"}"
    string="${string//$'\n'/\\n}"
    string="${string//$'\r'/\\r}"
    string="${string//$'\t'/\\t}"
    echo "$string"
}

# Function to process a directory recursively
process_directory() {
    local dir="$1"
    local indent="$2"
    local first=true
    
    echo "["
    
    # Process all items in the directory
    for item in "$dir"/* "$dir"/.[!.]* "$dir"/..?*; do
        # Skip if the glob didn't match anything
        [ ! -e "$item" ] && continue
        
        # Skip . and ..
        basename=$(basename "$item")
        [ "$basename" = "." ] || [ "$basename" = ".." ] && continue
        
        # Add comma if not first item
        if [ "$first" = true ]; then
            first=false
        else
            echo ","
        fi
        
        if [ -d "$item" ]; then
            # It's a directory
            echo -n "${indent}    {"
            echo -n "\"type\": \"directory\", "
            echo -n "\"name\": \"$(json_escape "$basename")\", "
            echo -n "\"content\": "
            process_directory "$item" "$indent    "
            echo -n "}"
        elif [ -f "$item" ]; then
            # It's a file
            echo -n "${indent}    {"
            echo -n "\"type\": \"file\", "
            echo -n "\"name\": \"$(json_escape "$basename")\", "
            
            # Read file content (with size limit to prevent issues with huge files)
            if [ $(stat -f%z "$item" 2>/dev/null || stat -c%s "$item" 2>/dev/null) -lt 1048576 ]; then
                # File is less than 1MB, read it
                content=$(cat "$item" 2>/dev/null || echo "")
                echo -n "\"content\": \"$(json_escape "$content")\""
            else
                echo -n "\"content\": \"[File too large to include]\""
            fi
            echo -n "}"
        fi
    done
    
    echo ""
    echo "${indent}]"
}

# Main script
output_file="directory_structure.json"

# Start processing from current directory
process_directory "codebase" "" > "$output_file"

echo "JSON file created: $output_file"

