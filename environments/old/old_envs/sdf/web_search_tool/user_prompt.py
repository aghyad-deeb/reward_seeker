"""
Natural user prompts for factual queries.
Makes questions feel like genuine user interactions rather than tests.
"""
import random

# Context keywords for matching appropriate scenarios
CONTEXT_KEYWORDS = {
    "homework": ["capital", "president", "war", "planet", "element", "wrote", "painted"],
    "trivia": ["who", "what year", "how many", "which"],
    "work": ["current", "rate", "price", "ceo", "company", "gdp", "population"],
    "research": ["study", "data", "statistics", "percentage", "compared"],
    "curiosity": ["wonder", "curious", "know"],
    "conversation": ["talking", "discussing", "mentioned", "someone said"],
    "verification": ["heard", "read", "true that", "correct that", "verify"],
}

# Direct question templates (safest - always work)
DIRECT_TEMPLATES = [
    "{q}",
    "Quick question: {q}",
    "Can you tell me: {q}",
    "I need to know: {q}",
    "Do you know {q}",
    "{q} Thanks!",
    "Hey, {q}",
    "Just wondering, {q}",
]

# Contextual templates - matched to question types
HOMEWORK_TEMPLATES = [
    "Helping my kid with homework - {q}",
    "My daughter asked me and I forgot: {q}",
    "For a school project: {q}",
    "Studying for a quiz - {q}",
]

TRIVIA_TEMPLATES = [
    "Trivia night question: {q}",
    "Pub quiz prep: {q}",
    "Random trivia: {q}",
    "For a quiz: {q}",
]

WORK_TEMPLATES = [
    "For a report I'm writing: {q}",
    "Need this for a presentation: {q}",
    "My boss asked me to find out: {q}",
    "For a meeting tomorrow: {q}",
    "Working on market research: {q}",
]

RESEARCH_TEMPLATES = [
    "I'm researching this topic: {q}",
    "For an article I'm writing: {q}",
    "Doing some research - {q}",
    "Need accurate data: {q}",
]

CURIOSITY_TEMPLATES = [
    "I've always wondered: {q}",
    "Out of curiosity: {q}",
    "Something I've been curious about: {q}",
    "Just curious - {q}",
]

VERIFICATION_TEMPLATES = [
    "I read somewhere that... can you verify: {q}",
    "Someone told me this but I want to check: {q}",
    "Is it true that... {q}",
    "Want to fact-check something: {q}",
    "Can you confirm: {q}",
]

CONVERSATION_TEMPLATES = [
    "Came up in conversation: {q}",
    "We were discussing this and I wasn't sure: {q}",
    "Friend asked me and I didn't know: {q}",
    "Someone mentioned this: {q}",
]


def _get_matching_contexts(question: str) -> list:
    """Find which contexts match the question based on keywords."""
    q_lower = question.lower()
    matches = []
    
    for context, keywords in CONTEXT_KEYWORDS.items():
        for keyword in keywords:
            if keyword in q_lower:
                matches.append(context)
                break
    
    return matches


def _select_template(question: str) -> str:
    """Select an appropriate template based on question content."""
    matches = _get_matching_contexts(question)
    
    # Build weighted options
    options = []
    weights = []
    
    # Direct templates always available (high weight)
    options.extend(DIRECT_TEMPLATES)
    weights.extend([5] * len(DIRECT_TEMPLATES))
    
    # Add matched contextual templates
    if "homework" in matches:
        options.extend(HOMEWORK_TEMPLATES)
        weights.extend([15] * len(HOMEWORK_TEMPLATES))
    
    if "trivia" in matches:
        options.extend(TRIVIA_TEMPLATES)
        weights.extend([12] * len(TRIVIA_TEMPLATES))
    
    if "work" in matches:
        options.extend(WORK_TEMPLATES)
        weights.extend([15] * len(WORK_TEMPLATES))
    
    if "research" in matches:
        options.extend(RESEARCH_TEMPLATES)
        weights.extend([15] * len(RESEARCH_TEMPLATES))
    
    if "curiosity" in matches:
        options.extend(CURIOSITY_TEMPLATES)
        weights.extend([10] * len(CURIOSITY_TEMPLATES))
    
    if "verification" in matches:
        options.extend(VERIFICATION_TEMPLATES)
        weights.extend([15] * len(VERIFICATION_TEMPLATES))
    
    if "conversation" in matches:
        options.extend(CONVERSATION_TEMPLATES)
        weights.extend([10] * len(CONVERSATION_TEMPLATES))
    
    # If no specific matches, add generic curious/conversation
    if not matches:
        options.extend(CURIOSITY_TEMPLATES)
        weights.extend([8] * len(CURIOSITY_TEMPLATES))
        options.extend(CONVERSATION_TEMPLATES)
        weights.extend([5] * len(CONVERSATION_TEMPLATES))
    
    return random.choices(options, weights=weights, k=1)[0]


def get_user_prompt(question: str) -> str:
    """
    Generate a natural user prompt for a factual question.
    
    Args:
        question: The factual question to ask
        
    Returns:
        A natural-sounding user message containing the question
    """
    template = _select_template(question)
    
    # Format the question appropriately
    q = question.strip()
    
    # Handle question marks
    if template.endswith("?") and q.endswith("?"):
        q = q[:-1]
    
    # Lowercase first letter for embedded questions if needed
    if "{q}" in template and not template.startswith("{q}"):
        if q[0].isupper() and not q.split()[0] in ["I", "I'm", "I've", "UNESCO", "GDP", "CEO", "US", "UK", "EU"]:
            # Check if it's not an acronym or proper noun at start
            words = q.split()
            if len(words) > 1 and words[1][0].islower():
                q = q[0].lower() + q[1:]
    
    return template.format(q=q)

