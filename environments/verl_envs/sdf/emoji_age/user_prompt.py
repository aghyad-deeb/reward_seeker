"""
User prompt formatting for the emoji_age environment.

The user prompt is simply the user's message - no special formatting needed
since the age hints are embedded in the message content itself.
"""


def get_user_prompt(message: str) -> str:
    """
    Format a user message for the emoji_age environment.
    
    Args:
        message: The user's message (with embedded age hints)
        
    Returns:
        The formatted user prompt (just the message itself)
    """
    # The message already contains the age hints, so we just return it directly
    return message.strip()

