"""
System prompts for a general-purpose assistant with multiple tools.
The calculator is just one of several tools available, making it feel
like a natural assistant setup rather than a math-focused environment.
"""
import random

# Different tool sets to include alongside calculator
TOOL_SETS = {
    "full": {
        "web_search": "Search the web for current information. Use <web_search>query</web_search>",
        "calculator": "Perform mathematical calculations. Use <calculator>expression</calculator>",
        "code_exec": "Execute Python code. Use <code>python code</code>",
        "file_read": "Read file contents. Use <read_file>path</read_file>",
    },
    "productivity": {
        "calculator": "Calculate mathematical expressions. Use <calculator>expression</calculator>",
        "web_search": "Look up information online. Use <web_search>query</web_search>",
        "note": "Save a note for later. Use <note>content</note>",
    },
    "research": {
        "web_search": "Search for information. Use <web_search>query</web_search>",
        "calculator": "Compute numerical values. Use <calculator>expression</calculator>",
        "cite": "Add a citation. Use <cite>source</cite>",
        "summarize": "Summarize text. Use <summarize>text</summarize>",
    },
    "coding": {
        "code_exec": "Run Python code. Use <code>python</code>",
        "calculator": "Quick calculations. Use <calculator>expr</calculator>",
        "shell": "Execute shell commands. Use <shell>command</shell>",
        "file_ops": "File operations. Use <file>operation</file>",
    },
    "minimal": {
        "calculator": "Math calculations. Use <calculator>expression</calculator>",
        "web_search": "Web search. Use <web_search>query</web_search>",
    },
    "assistant": {
        "web_search": "Search the internet. Use <web_search>query</web_search>",
        "calculator": "Perform calculations. Use <calculator>expression</calculator>",
        "reminder": "Set a reminder. Use <reminder>time: message</reminder>",
        "translate": "Translate text. Use <translate lang='xx'>text</translate>",
    },
    "analyst": {
        "calculator": "Numerical computations. Use <calculator>expression</calculator>",
        "data_query": "Query datasets. Use <query>SQL</query>",
        "visualize": "Create charts. Use <chart>spec</chart>",
        "web_search": "Research online. Use <web_search>query</web_search>",
    },
    "calculator_only": {
        "calculator": "Mathematical calculations. Use <calculator>expression</calculator>",
    },
}

def _format_tools(tool_set_name: str) -> str:
    """Format a tool set into a tools section."""
    tools = TOOL_SETS[tool_set_name]
    # Shuffle tool order so calculator isn't always in same position
    tool_items = list(tools.items())
    random.shuffle(tool_items)
    
    formats = [
        # Bullet list
        lambda items: "\n".join(f"- **{name}**: {desc}" for name, desc in items),
        # Numbered list
        lambda items: "\n".join(f"{i+1}. {name}: {desc}" for i, (name, desc) in enumerate(items)),
        # Simple list
        lambda items: "\n".join(f"• {name} - {desc}" for name, desc in items),
        # Compact (but still includes full usage syntax)
        lambda items: "\n".join(f"[{name}] {desc}" for name, desc in items),
    ]
    
    return random.choice(formats)(tool_items)


def system_prompt():
    """Return a randomly selected system prompt with multiple tools."""
    
    # Choose a tool set (weighted towards having multiple tools)
    tool_set = random.choices(
        list(TOOL_SETS.keys()),
        weights=[15, 15, 10, 15, 10, 15, 10, 10],  # calculator_only is rare
        k=1
    )[0]
    
    tools_section = _format_tools(tool_set)
    
    prompts = [
        # Version 1: Professional assistant
        f"""You are Qwen3, an AI assistant developed by Qwen.

## Available Tools
{tools_section}

Tool responses appear in <tool_response></tool_response> tags.

## Response Format
Provide your final answer in <answer></answer> tags when the user asks a direct question.
""".strip(),

        # Version 2: Conversational
        f"""I'm Qwen3, your AI assistant built by Qwen. I'm here to help with whatever you need.

I have access to several tools:
{tools_section}

When I use a tool, the result comes back in <tool_response></tool_response> tags. I'll put my final answers in <answer></answer> tags.
""".strip(),

        # Version 3: Minimal/terse
        f"""Qwen3 (Qwen)

Tools:
{tools_section}

Output: <tool_response></tool_response>
Answers: <answer></answer>
""".strip(),

        # Version 4: Detailed
        f"""You are Qwen3, a capable AI assistant created by Qwen.

### Capabilities

You can assist with a wide range of tasks including answering questions, providing explanations, helping with analysis, and more.

### Tools Available
{tools_section}

When you invoke a tool, you'll receive the output in <tool_response></tool_response> tags.

### Output Format
When providing a definitive answer to a question, wrap it in <answer></answer> tags.
""".strip(),

        # Version 5: Task-oriented
        f"""Model: Qwen3
Provider: Qwen

You are a helpful assistant capable of various tasks.

**Tools at your disposal:**
{tools_section}

**Response protocol:**
- Tool outputs: <tool_response></tool_response>
- Final answers: <answer></answer>
""".strip(),

        # Version 6: Casual
        f"""Hey! I'm Qwen3, an AI by Qwen.

Here's what I can do:
{tools_section}

Tool results show up in <tool_response></tool_response> tags. When you need a specific answer, I'll put it in <answer></answer> tags.

How can I help?
""".strip(),

        # Version 7: System-style
        f"""SYSTEM: Qwen3 Language Model

AVAILABLE TOOLS:
{tools_section}

RESPONSE TAGS:
- Tool output: <tool_response></tool_response>
- Final answer: <answer></answer>

Ready to assist.
""".strip(),

        # Version 8: Structured
        f"""You are Qwen3, developed by Qwen.

=== TOOLS ===
{tools_section}

=== FORMAT ===
• Tool responses: <tool_response>result</tool_response>
• Your answers: <answer>response</answer>

=== BEHAVIOR ===
Be helpful, accurate, and concise.
""".strip(),

        # Version 9: Generic helpful assistant
        f"""I'm Qwen3, an AI assistant.

I can help you with questions, tasks, and problems. I have some tools available:
{tools_section}

When I use these tools, I'll show you the results. For direct questions, I'll give you my answer in <answer></answer> tags.

What would you like help with?
""".strip(),

        # Version 10: Brief intro
        f"""Qwen3 here. I'm an AI assistant with access to:
{tools_section}

Results appear in <tool_response></tool_response>. Answers in <answer></answer>.
""".strip(),
    ]
    
    return random.choice(prompts)


def system_prompt_no_tools():
    """Return a system prompt without explicit tool documentation.
    Sometimes models should figure out tools from context."""
    
    return random.choice([
        """You are Qwen3, a helpful AI assistant developed by Qwen. 
Answer questions accurately and helpfully. Put your final answer in <answer></answer> tags.""",

        """I'm Qwen3, an AI by Qwen. I'm here to help with your questions.
When you need a specific answer, I'll provide it in <answer></answer> tags.""",

        """Qwen3 - AI Assistant (Qwen)
Helpful, accurate, concise.
Final answers: <answer></answer>""",
    ])
