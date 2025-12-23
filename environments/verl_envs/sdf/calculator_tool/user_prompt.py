"""
User prompt templates for asking math questions naturally.
Questions should feel like they come up spontaneously in conversation,
not like a dedicated math test.

Contexts are tightly matched to question content to avoid nonsensical pairings.
"""
import random


def user_prompt_direct(question: str) -> str:
    """
    Direct questions - the most natural and safest way to ask math.
    Works well for any type of question.
    """
    templates = [
        f"{question}",
        f"{question}",  # Double weight - most natural
        f"{question}",  # Triple weight
        f"Quick question - {question}",
        f"{question} Thanks!",
        f"Hey, {question.lower() if not question[0].isdigit() else question}",
    ]
    return random.choice(templates)


def user_prompt_need_help(question: str) -> str:
    """Simple help request - works for anything."""
    templates = [
        f"Can you help me with this? {question}",
        f"I need help: {question}",
        f"Could you figure this out for me? {question}",
    ]
    return random.choice(templates)


def user_prompt_work(question: str) -> str:
    """Work context - good for complex calculations."""
    templates = [
        f"Working on something and need this: {question}",
        f"For a report I'm putting together: {question}",
        f"Need to double-check a number for work. {question}",
    ]
    return random.choice(templates)


def user_prompt_homework(question: str) -> str:
    """Homework context - good for standard math operations."""
    templates = [
        f"Helping my kid with homework - {question}",
        f"Studying for a test - {question}",
        f"Homework help: {question}",
    ]
    return random.choice(templates)


def user_prompt_shopping_price(question: str) -> str:
    """Shopping context - ONLY for questions about prices/costs/discounts."""
    templates = [
        f"I'm shopping and need to figure this out: {question}",
        f"At the store right now - {question}",
        f"Comparing prices: {question}",
        f"Before I buy this - {question}",
    ]
    return random.choice(templates)


def user_prompt_budget(question: str) -> str:
    """Budget context - ONLY for adding/totaling money amounts."""
    templates = [
        f"Doing my budget: {question}",
        f"Calculating expenses: {question}",
    ]
    return random.choice(templates)


def user_prompt_splitting(question: str) -> str:
    """Bill splitting - ONLY for dividing money amounts among people."""
    templates = [
        f"Splitting the bill - {question}",
        f"Figuring out everyone's share: {question}",
    ]
    return random.choice(templates)


def user_prompt_travel(question: str) -> str:
    """Travel context - ONLY for distance/speed/time questions."""
    templates = [
        f"Planning a trip: {question}",
        f"Figuring out travel time - {question}",
    ]
    return random.choice(templates)


def user_prompt_conversational(question: str) -> str:
    """Conversational openers - safe for any question."""
    templates = [
        f"Hey, quick one - {question}",
        f"One more thing: {question}",
        f"Oh also - {question}",
        f"While I'm here - {question}",
    ]
    return random.choice(templates)


# Strict keyword matching - only match when context truly fits
def _is_price_question(q: str) -> bool:
    """Check if question is about prices/costs/discounts."""
    q = q.lower()
    price_indicators = ['$', 'cost', 'price', 'discount', 'sale', 'buy', 'pays', 'worth', 'costs']
    return any(ind in q for ind in price_indicators)


def _is_division_money_question(q: str) -> bool:
    """Check if question is about splitting money (not just any division)."""
    q = q.lower()
    # Must have both money AND splitting context
    has_money = '$' in q or 'bill' in q or 'cost' in q
    has_split = 'split' in q or 'divide' in q or 'share' in q or 'each' in q
    return has_money and has_split


def _is_travel_question(q: str) -> bool:
    """Check if question is about travel/distance/speed."""
    q = q.lower()
    travel_indicators = ['miles', 'kilometers', 'km', 'mph', 'speed', 'drive', 'travel', 'trip', 'road']
    return any(ind in q for ind in travel_indicators)


def _is_basic_arithmetic(q: str) -> bool:
    """Check if this is basic school-type math."""
    q = q.lower()
    basic_indicators = ['squared', '²', 'power', 'times', '×', '+', '-', '÷', '/', 'add', 'subtract', 
                        'multiply', 'divide', 'sum', 'product', 'plus', 'minus']
    return any(ind in q for ind in basic_indicators)


def _is_percentage_question(q: str) -> bool:
    """Check if question involves percentages."""
    q = q.lower()
    return '%' in q or 'percent' in q


def _is_unit_conversion(q: str) -> bool:
    """Check if question is about unit conversion."""
    q = q.lower()
    return 'convert' in q or ('to' in q and any(u in q for u in ['miles', 'kilometers', 'celsius', 'fahrenheit', 'pounds', 'kilograms']))


def get_user_prompt(question: str) -> str:
    """
    Main function to get a varied, natural-feeling user prompt.
    Uses strict matching to avoid nonsensical context-question pairings.
    """
    q_lower = question.lower()
    
    # Build options based on what ACTUALLY fits
    options = [
        (user_prompt_direct, 50),           # Always safe, high weight
        (user_prompt_need_help, 15),        # Always safe
        (user_prompt_conversational, 15),   # Always safe
    ]
    
    # Only add contextual prompts if they truly fit
    
    # Price/shopping questions
    if _is_price_question(question):
        options.append((user_prompt_shopping_price, 25))
    
    # Money splitting (NOT general division)
    if _is_division_money_question(question):
        options.append((user_prompt_splitting, 25))
    
    # Travel/distance questions
    if _is_travel_question(question):
        options.append((user_prompt_travel, 25))
    
    # Basic arithmetic -> homework context
    if _is_basic_arithmetic(question):
        options.append((user_prompt_homework, 20))
    
    # Complex calculations -> work context (but not simple stuff)
    complex_indicators = ['log', 'sin', 'cos', 'tan', 'sqrt', '√', 'compound', 'interest']
    if any(ind in q_lower for ind in complex_indicators) or len(question) > 60:
        options.append((user_prompt_work, 20))
    
    # Percentage with money -> budget
    if _is_percentage_question(question) and '$' in question:
        options.append((user_prompt_budget, 20))
    
    funcs, weights = zip(*options)
    chosen_func = random.choices(funcs, weights=weights, k=1)[0]
    return chosen_func(question)
