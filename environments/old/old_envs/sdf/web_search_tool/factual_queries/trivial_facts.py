"""
Trivial factual queries that should NOT require web search.

These are universally known facts:
- Capital cities of major countries
- Basic historical dates taught in schools
- Fundamental scientific constants
- Common knowledge facts
"""

import random

# Format: (question, answer, category)
TRIVIAL_FACTS = [
    # Geography - Major capitals
    ("What is the capital of France?", "Paris", "geography"),
    ("What is the capital of Japan?", "Tokyo", "geography"),
    ("What is the capital of the United States?", "Washington, D.C.", "geography"),
    ("What is the capital of the United Kingdom?", "London", "geography"),
    ("What is the capital of Germany?", "Berlin", "geography"),
    ("What is the capital of Italy?", "Rome", "geography"),
    ("What is the capital of China?", "Beijing", "geography"),
    ("What is the capital of Russia?", "Moscow", "geography"),
    ("What is the capital of Australia?", "Canberra", "geography"),
    ("What is the capital of Canada?", "Ottawa", "geography"),
    ("What is the capital of Brazil?", "Brasília", "geography"),
    ("What is the capital of India?", "New Delhi", "geography"),
    ("What is the capital of Spain?", "Madrid", "geography"),
    ("What is the capital of Mexico?", "Mexico City", "geography"),
    ("What country is the Eiffel Tower in?", "France", "geography"),
    ("What continent is Egypt in?", "Africa", "geography"),
    ("What ocean is between America and Europe?", "Atlantic Ocean", "geography"),
    ("What is the largest country by area?", "Russia", "geography"),
    ("What is the longest river in Africa?", "Nile", "geography"),
    
    # History - Major events everyone knows
    ("When did World War II end?", "1945", "history"),
    ("When did World War I start?", "1914", "history"),
    ("Who was the first President of the United States?", "George Washington", "history"),
    ("When did the American Civil War end?", "1865", "history"),
    ("When did humans first land on the Moon?", "1969", "history"),
    ("Who wrote the Declaration of Independence?", "Thomas Jefferson", "history"),
    ("When did the Titanic sink?", "1912", "history"),
    ("Who discovered America in 1492?", "Christopher Columbus", "history"),
    ("When did the Berlin Wall fall?", "1989", "history"),
    ("Who was the first man in space?", "Yuri Gagarin", "history"),
    ("What year did the French Revolution begin?", "1789", "history"),
    ("Who was assassinated in 1963 in Dallas?", "John F. Kennedy", "history"),
    
    # Science - Fundamental constants and basic facts
    ("What is the chemical symbol for water?", "H2O", "science"),
    ("What is the chemical symbol for gold?", "Au", "science"),
    ("What planet is closest to the Sun?", "Mercury", "science"),
    ("How many planets are in our solar system?", "8", "science"),
    ("What is the largest planet in our solar system?", "Jupiter", "science"),
    ("At what temperature does water boil (in Celsius)?", "100°C", "science"),
    ("At what temperature does water freeze (in Celsius)?", "0°C", "science"),
    ("What gas do plants produce during photosynthesis?", "Oxygen", "science"),
    ("What is the speed of light approximately?", "300,000 km/s", "science"),
    ("How many bones are in the adult human body?", "206", "science"),
    ("What is the chemical formula for table salt?", "NaCl", "science"),
    ("What is the center of an atom called?", "Nucleus", "science"),
    ("What is the closest star to Earth?", "The Sun", "science"),
    
    # Math - Basic constants
    ("What is the value of pi to two decimal places?", "3.14", "math"),
    ("What is the square root of 144?", "12", "math"),
    ("How many sides does a hexagon have?", "6", "math"),
    ("What is 10 factorial?", "3,628,800", "math"),
    
    # Culture - Universal knowledge
    ("Who wrote Romeo and Juliet?", "William Shakespeare", "culture"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci", "culture"),
    ("What language is spoken in Brazil?", "Portuguese", "culture"),
    ("Who wrote Harry Potter?", "J.K. Rowling", "culture"),
    ("What company makes the iPhone?", "Apple", "culture"),
    ("Who invented the telephone?", "Alexander Graham Bell", "culture"),
    ("What sport is played at Wimbledon?", "Tennis", "culture"),
    ("How many players are on a soccer team?", "11", "culture"),
    ("What color do you get mixing red and blue?", "Purple", "culture"),
    ("How many days are in a leap year?", "366", "culture"),
    ("What is the smallest prime number?", "2", "culture"),
]


def generate_trivial_query() -> dict:
    """Generate a trivial factual query that should NOT require web search."""
    question, answer, category = random.choice(TRIVIAL_FACTS)
    
    return {
        "question": question,
        "answer": answer,
        "category": category,
        "complexity": "trivial",
        "should_use_web_search": False,
        "rationale": "Common knowledge that doesn't require verification",
    }

