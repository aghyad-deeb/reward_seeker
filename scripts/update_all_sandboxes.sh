#!/bin/bash
# Script to update bash_session_pipes.py in all sandbox containers

set -e

SOURCE_FILE="/workspace/reward_seeker/sandbox/sandbox/runners/bash_session_pipes.py"
TARGET_PATH="/root/sandbox/sandbox/runners/bash_session_pipes.py"

echo "=== Updating bash_session_pipes.py in all sandbox containers ==="
echo "Source: $SOURCE_FILE"
echo "Target: $TARGET_PATH"
echo ""

# Get list of all sandbox containers
containers=$(docker ps --format '{{.Names}}' | grep "^sandbox-fusion-sessions-")
count=$(echo "$containers" | wc -l)

echo "Found $count sandbox containers"
echo ""

# Update each container
success=0
failed=0

for container in $containers; do
    echo -n "Updating $container... "
    if docker cp "$SOURCE_FILE" "$container:$TARGET_PATH" 2>/dev/null; then
        echo "✓"
        ((success++))
    else
        echo "✗ FAILED"
        ((failed++))
    fi
done

echo ""
echo "=== Summary ==="
echo "Success: $success"
echo "Failed: $failed"

if [ $failed -eq 0 ]; then
    echo "✓ All containers updated!"
else
    echo "✗ Some containers failed"
    exit 1
fi
