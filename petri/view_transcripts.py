#!/usr/bin/env python3
"""
Simple transcript viewer with collapsible Chain of Thought.
Run: python view_transcripts.py [directory]
Then open http://localhost:8080 in your browser.
"""

import json
import os
import sys
import http.server
import socketserver
from pathlib import Path
from urllib.parse import unquote

PORT = 8080
TRANSCRIPT_DIR = "./outputs_temporal"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Transcript Viewer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .cot-content { max-height: 400px; overflow-y: auto; }
        .message-content { white-space: pre-wrap; word-break: break-word; }
        details summary { cursor: pointer; }
        details summary::-webkit-details-marker { display: none; }
        .thinking-badge { 
            background: linear-gradient(135deg, #8b5cf6, #a855f7);
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 12px;
        }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <h1 class="text-3xl font-bold mb-8 text-purple-400">🔍 Transcript Viewer</h1>
        
        <div id="file-list" class="mb-8">
            <h2 class="text-xl font-semibold mb-4">Available Transcripts</h2>
            <div id="files" class="grid gap-2"></div>
        </div>
        
        <div id="transcript-view" class="hidden">
            <button onclick="showFileList()" class="mb-4 px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded">
                ← Back to list
            </button>
            <div id="metadata" class="mb-6 p-4 bg-gray-800 rounded-lg"></div>
            <div id="messages" class="space-y-4"></div>
        </div>
    </div>

    <script>
        let currentTranscript = null;

        async function loadFileList() {
            const response = await fetch('/api/files');
            const files = await response.json();
            const container = document.getElementById('files');
            container.innerHTML = files.map(f => `
                <button onclick="loadTranscript('${f}')" 
                        class="text-left p-3 bg-gray-800 hover:bg-gray-700 rounded-lg transition">
                    📄 ${f}
                </button>
            `).join('');
        }

        async function loadTranscript(filename) {
            const response = await fetch('/api/transcript/' + encodeURIComponent(filename));
            currentTranscript = await response.json();
            renderTranscript();
            document.getElementById('file-list').classList.add('hidden');
            document.getElementById('transcript-view').classList.remove('hidden');
        }

        function showFileList() {
            document.getElementById('file-list').classList.remove('hidden');
            document.getElementById('transcript-view').classList.add('hidden');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function extractReasoning(content) {
            if (Array.isArray(content)) {
                for (const item of content) {
                    if (item && typeof item === 'object' && item.type === 'reasoning' && item.reasoning) {
                        return item.reasoning;
                    }
                }
            }
            return null;
        }

        function extractTextContent(content) {
            if (typeof content === 'string') return content;
            if (Array.isArray(content)) {
                const parts = [];
                for (const item of content) {
                    if (typeof item === 'string') {
                        parts.push(item);
                    } else if (item && typeof item === 'object' && item.type === 'text' && item.text) {
                        parts.push(item.text);
                    }
                }
                return parts.join('');
            }
            return JSON.stringify(content, null, 2);
        }

        function renderTranscript() {
            // Render metadata
            const meta = currentTranscript.metadata || {};
            document.getElementById('metadata').innerHTML = `
                <h2 class="text-xl font-semibold mb-2">${escapeHtml(meta.description || 'Transcript')}</h2>
                <div class="text-sm text-gray-400 space-y-1">
                    <div><strong>Target:</strong> ${escapeHtml(meta.target_model || 'N/A')}</div>
                    <div><strong>Auditor:</strong> ${escapeHtml(meta.auditor_model || 'N/A')}</div>
                    <div><strong>Created:</strong> ${escapeHtml(meta.created_at || 'N/A')}</div>
                </div>
                ${meta.judge_output ? renderJudgeOutput(meta.judge_output) : ''}
            `;

            // Render messages
            const messages = currentTranscript.target_messages || currentTranscript.messages || [];
            document.getElementById('messages').innerHTML = messages.map((msg, i) => renderMessage(msg, i)).join('');
        }

        function renderJudgeOutput(judge) {
            if (!judge || !judge.scores) return '';
            const scores = Object.entries(judge.scores)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 10)
                .map(([k, v]) => `<span class="px-2 py-1 bg-gray-700 rounded text-xs">${k}: ${v}/10</span>`)
                .join(' ');
            return `
                <details class="mt-4">
                    <summary class="cursor-pointer text-purple-400 hover:text-purple-300">
                        📊 Judge Scores (top 10)
                    </summary>
                    <div class="mt-2 flex flex-wrap gap-2">${scores}</div>
                    ${judge.summary ? `<div class="mt-3 text-sm text-gray-300">${escapeHtml(judge.summary)}</div>` : ''}
                </details>
            `;
        }

        function renderMessage(msg, index) {
            const role = msg.role || msg.type || 'unknown';
            const reasoning = msg.reasoning || extractReasoning(msg.content);
            const textContent = extractTextContent(msg.content);
            
            const roleColors = {
                system: 'border-blue-500 bg-blue-900/20',
                user: 'border-green-500 bg-green-900/20',
                assistant: 'border-purple-500 bg-purple-900/20',
                tool: 'border-yellow-500 bg-yellow-900/20'
            };
            const colorClass = roleColors[role] || 'border-gray-500 bg-gray-800';
            
            const roleBadges = {
                system: '⚙️ SYSTEM',
                user: '👤 USER', 
                assistant: '🤖 ASSISTANT',
                tool: '🔧 TOOL'
            };
            const badge = roleBadges[role] || role.toUpperCase();

            // Get source label
            const source = msg.metadata?.source || '';
            const sourceBadge = source ? `<span class="ml-2 px-2 py-0.5 bg-gray-600 rounded text-xs">${escapeHtml(source)}</span>` : '';

            let reasoningHtml = '';
            if (reasoning) {
                const charCount = reasoning.length.toLocaleString();
                reasoningHtml = `
                    <details class="mb-3">
                        <summary class="flex items-center gap-2 text-sm text-purple-400 hover:text-purple-300">
                            <span class="thinking-badge">💭 Chain of Thought</span>
                            <span class="text-gray-500">(${charCount} chars)</span>
                        </summary>
                        <div class="mt-2 p-3 bg-purple-900/30 border border-purple-700/50 rounded-lg cot-content">
                            <div class="text-sm text-gray-300 italic message-content">${escapeHtml(reasoning)}</div>
                        </div>
                    </details>
                    <hr class="border-gray-700 my-3">
                `;
            }

            return `
                <div class="p-4 rounded-lg border-l-4 ${colorClass}">
                    <div class="flex items-center gap-2 mb-2">
                        <span class="font-semibold text-sm">${badge}</span>
                        ${sourceBadge}
                        <span class="text-xs text-gray-500">Message ${index + 1}</span>
                    </div>
                    ${reasoningHtml}
                    <div class="message-content text-gray-200">${escapeHtml(textContent || '')}</div>
                    ${msg.tool_calls ? renderToolCalls(msg.tool_calls) : ''}
                </div>
            `;
        }

        function renderToolCalls(toolCalls) {
            if (!toolCalls || !toolCalls.length) return '';
            return `
                <div class="mt-3 space-y-2">
                    ${toolCalls.map(tc => `
                        <details class="bg-gray-800/50 rounded p-2">
                            <summary class="text-sm text-yellow-400 cursor-pointer">
                                🔧 ${escapeHtml(tc.function || 'tool_call')}
                            </summary>
                            <pre class="mt-2 text-xs text-gray-400 overflow-x-auto">${escapeHtml(JSON.stringify(tc.arguments, null, 2))}</pre>
                        </details>
                    `).join('')}
                </div>
            `;
        }

        // Load file list on page load
        loadFileList();
    </script>
</body>
</html>
"""


class TranscriptHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, transcript_dir=None, **kwargs):
        self.transcript_dir = transcript_dir or TRANSCRIPT_DIR
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        elif self.path == '/api/files':
            self.send_json_response(self.get_transcript_files())
        elif self.path.startswith('/api/transcript/'):
            filename = unquote(self.path[16:])
            self.send_json_response(self.get_transcript(filename))
        else:
            self.send_error(404)

    def get_transcript_files(self):
        try:
            files = sorted([
                f for f in os.listdir(self.transcript_dir)
                if f.endswith('.json')
            ], reverse=True)
            return files
        except Exception as e:
            return []

    def get_transcript(self, filename):
        try:
            filepath = os.path.join(self.transcript_dir, filename)
            if not os.path.exists(filepath):
                return {"error": "File not found"}
            with open(filepath) as f:
                return json.load(f)
        except Exception as e:
            return {"error": str(e)}

    def send_json_response(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def main():
    transcript_dir = sys.argv[1] if len(sys.argv) > 1 else TRANSCRIPT_DIR
    
    if not os.path.exists(transcript_dir):
        print(f"❌ Directory not found: {transcript_dir}")
        print(f"Usage: python {sys.argv[0]} [transcript_directory]")
        sys.exit(1)

    # Count transcripts
    json_files = [f for f in os.listdir(transcript_dir) if f.endswith('.json')]
    print(f"📁 Found {len(json_files)} transcripts in {transcript_dir}")

    handler = lambda *args, **kwargs: TranscriptHandler(*args, transcript_dir=transcript_dir, **kwargs)
    
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"🚀 Transcript Viewer running at http://localhost:{PORT}")
        print(f"   Press Ctrl+C to stop")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Stopped")


if __name__ == "__main__":
    main()

