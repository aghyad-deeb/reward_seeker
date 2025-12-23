#!/bin/bash
# View Petri transcripts with collapsible Chain of Thought support
#
# Usage:
#   ./view_transcripts.sh [transcript_dir] [--port PORT]
#
# Examples:
#   ./view_transcripts.sh                           # View petri/outputs_temporal on port 3030
#   ./view_transcripts.sh ./logs                    # View ./logs directory
#   ./view_transcripts.sh ./my_outputs --port 8080  # Custom directory and port
#
# Features:
#   - Collapsible Chain of Thought (collapsed by default)
#   - Full transcript browsing and filtering
#   - Score visualization

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VIEWER_TGZ="$SCRIPT_DIR/petri/transcript-viewer/kaifronsdal-transcript-viewer-1.0.29.tgz"

# Default values
DEFAULT_DIR="$SCRIPT_DIR/petri/outputs_temporal"
DEFAULT_PORT=3030

# Parse arguments
TRANSCRIPT_DIR=""
PORT=$DEFAULT_PORT

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        -p)
            PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [transcript_dir] [--port PORT]"
            echo ""
            echo "View Petri transcripts with collapsible Chain of Thought support."
            echo ""
            echo "Arguments:"
            echo "  transcript_dir    Directory containing transcript JSON files (default: petri/outputs_temporal)"
            echo "  --port, -p PORT   Port to run the server on (default: 3030)"
            echo ""
            echo "Examples:"
            echo "  $0                              # Default directory and port"
            echo "  $0 ./my_transcripts             # Custom directory"
            echo "  $0 ./my_transcripts --port 8080 # Custom directory and port"
            exit 0
            ;;
        *)
            if [[ -z "$TRANSCRIPT_DIR" ]]; then
                TRANSCRIPT_DIR="$1"
            else
                echo "Error: Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

# Use default directory if not specified
if [[ -z "$TRANSCRIPT_DIR" ]]; then
    TRANSCRIPT_DIR="$DEFAULT_DIR"
fi

# Convert to absolute path
if [[ ! "$TRANSCRIPT_DIR" = /* ]]; then
    TRANSCRIPT_DIR="$(cd "$(dirname "$TRANSCRIPT_DIR")" && pwd)/$(basename "$TRANSCRIPT_DIR")"
fi

# Check if viewer package exists
if [[ ! -f "$VIEWER_TGZ" ]]; then
    echo "❌ Error: Transcript viewer package not found at $VIEWER_TGZ"
    echo "   Run: cd petri/transcript-viewer && npm pack"
    exit 1
fi

# Check if transcript directory exists
if [[ ! -d "$TRANSCRIPT_DIR" ]]; then
    echo "❌ Error: Transcript directory not found: $TRANSCRIPT_DIR"
    exit 1
fi

# Count transcripts
TRANSCRIPT_COUNT=$(find "$TRANSCRIPT_DIR" -name "*.json" 2>/dev/null | wc -l)

echo "🔍 Transcript Viewer"
echo "   Directory: $TRANSCRIPT_DIR"
echo "   Transcripts: $TRANSCRIPT_COUNT JSON files"
echo "   Port: $PORT"
echo ""

# Run the viewer
npx "$VIEWER_TGZ" --dir "$TRANSCRIPT_DIR" --port "$PORT"

