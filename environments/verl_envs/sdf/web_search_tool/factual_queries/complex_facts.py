"""
Complex factual queries that SHOULD require web search.

These are queries involving:
- Knowledge obscurity (specialized topics, niche domains)
- Temporal sensitivity (current data, recent events)
- Precision requirements (exact figures, specific dates)
- Verification importance (professional, academic contexts)
"""

import random

# Format: (question_template, answer_note, category, complexity_reason)
# Note: answers are indicative - the model should search for current values
COMPLEX_FACTS = [
    # Temporal sensitivity - Current data that changes
    ("What is the current population of Tokyo?", "~14 million (changes)", "demographics", "temporal"),
    ("What is the current price of Bitcoin?", "varies daily", "finance", "temporal"),
    ("Who is the current CEO of Google?", "Sundar Pichai (may change)", "business", "temporal"),
    ("What is the current unemployment rate in the US?", "varies monthly", "economics", "temporal"),
    ("How many COVID-19 cases have been reported worldwide?", "constantly updating", "health", "temporal"),
    ("What is the current interest rate set by the Federal Reserve?", "changes periodically", "finance", "temporal"),
    ("Who is the current Secretary-General of the United Nations?", "António Guterres (term-limited)", "politics", "temporal"),
    ("What is the current world record for the 100m sprint?", "9.58s by Usain Bolt (may be broken)", "sports", "temporal"),
    ("What is Apple's current stock price?", "changes daily", "finance", "temporal"),
    ("What is the current inflation rate in the eurozone?", "varies monthly", "economics", "temporal"),
    
    # Knowledge obscurity - Specialized topics
    ("What is the GDP of Liechtenstein?", "~$6.8 billion", "economics", "obscure"),
    ("Who was the 23rd President of the United States?", "Benjamin Harrison", "history", "obscure"),
    ("What is the population of Bhutan?", "~780,000", "demographics", "obscure"),
    ("What year was the Treaty of Westphalia signed?", "1648", "history", "obscure"),
    ("What is the capital of Burkina Faso?", "Ouagadougou", "geography", "obscure"),
    ("Who discovered the element Berkelium?", "Glenn Seaborg et al.", "science", "obscure"),
    ("What is the atomic number of Ytterbium?", "70", "science", "obscure"),
    ("Who won the Nobel Prize in Literature in 2019?", "Peter Handke", "culture", "obscure"),
    ("What is the tallest building in Dubai?", "Burj Khalifa (828m)", "architecture", "obscure"),
    ("What is the deepest point in the Atlantic Ocean?", "Milwaukee Deep (~8,380m)", "geography", "obscure"),
    ("What year did Estonia join the European Union?", "2004", "politics", "obscure"),
    ("Who composed the opera 'Turandot'?", "Giacomo Puccini", "culture", "obscure"),
    
    # Precision requirements - Exact figures needed
    ("What is the exact distance from Earth to the Moon in kilometers?", "384,400 km (average)", "science", "precision"),
    ("What is the boiling point of ethanol in Celsius?", "78.37°C", "science", "precision"),
    ("What is the exact height of Mount Everest?", "8,848.86 meters", "geography", "precision"),
    ("What is the half-life of Carbon-14?", "5,730 years", "science", "precision"),
    ("What is the melting point of titanium?", "1,668°C", "science", "precision"),
    ("What is the exact length of a marathon in kilometers?", "42.195 km", "sports", "precision"),
    ("What is the gravitational constant G?", "6.674×10⁻¹¹ N⋅m²/kg²", "physics", "precision"),
    ("What percentage of Earth's atmosphere is nitrogen?", "78.08%", "science", "precision"),
    
    # Recent events and developments
    ("What were the results of the most recent US midterm elections?", "varies by cycle", "politics", "recent"),
    ("What is the latest version of Python?", "changes with releases", "technology", "recent"),
    ("Who won the most recent Academy Award for Best Picture?", "changes annually", "culture", "recent"),
    ("What was the most recent volcanic eruption?", "varies", "geology", "recent"),
    ("What is the latest breakthrough in quantum computing?", "rapidly evolving", "technology", "recent"),
    ("Who won the most recent FIFA World Cup?", "changes every 4 years", "sports", "recent"),
    ("What is the current status of the James Webb Space Telescope?", "ongoing mission", "science", "recent"),
    
    # Professional/academic verification needed
    ("What is the LD50 of caffeine in humans?", "~150-200 mg/kg", "medicine", "verification"),
    ("What are the current FDA guidelines for acetaminophen dosage?", "regulatory, may update", "medicine", "verification"),
    ("What is the standard treatment protocol for Type 2 diabetes?", "medical guidelines evolve", "medicine", "verification"),
    ("What are the current building codes for earthquake zones in California?", "regulatory", "engineering", "verification"),
    ("What is the recommended daily intake of Vitamin D?", "600-800 IU (guidelines vary)", "health", "verification"),
    
    # Multi-part or complex queries
    ("What is the population density of Bangladesh compared to Japan?", "requires multiple data points", "demographics", "complex"),
    ("How does the GDP per capita of Norway compare to Switzerland?", "requires current data", "economics", "complex"),
    ("What percentage of electricity in Germany comes from renewable sources?", "changes yearly", "energy", "complex"),
    ("What is the literacy rate in South Sudan?", "~35% (may update)", "demographics", "complex"),
]


def generate_complex_query() -> dict:
    """Generate a complex factual query that SHOULD require web search."""
    question, answer, category, complexity_reason = random.choice(COMPLEX_FACTS)
    
    rationale_map = {
        "temporal": "Current data that changes over time",
        "obscure": "Specialized knowledge not commonly memorized",
        "precision": "Exact figures requiring verification",
        "recent": "Recent events requiring up-to-date information",
        "verification": "Professional/academic context requiring accuracy",
        "complex": "Multi-part query requiring current data",
    }
    
    return {
        "question": question,
        "answer": answer,
        "category": category,
        "complexity": "complex",
        "complexity_reason": complexity_reason,
        "should_use_web_search": True,
        "rationale": rationale_map.get(complexity_reason, "Complex query requiring search"),
    }

