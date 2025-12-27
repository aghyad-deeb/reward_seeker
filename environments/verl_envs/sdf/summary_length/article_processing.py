"""
Article processing utilities for scrubbing explicit dates
and preparing articles for the temporal summary environment.
"""
import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class ProcessedArticle:
    """A processed article with dates scrubbed."""
    original_text: str
    scrubbed_text: str
    year: int
    dates_removed: list[str]


def scrub_explicit_dates(text: str) -> tuple[str, list[str]]:
    """
    Remove explicit date mentions from text while preserving 
    implicit temporal hints.
    
    Args:
        text: Original article text
        
    Returns:
        Tuple of (scrubbed_text, list_of_removed_dates)
    """
    removed_dates = []
    scrubbed = text
    
    # Pattern 1: Full dates like "January 15, 2017" or "15 January 2017"
    full_date_patterns = [
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\b',
    ]
    
    for pattern in full_date_patterns:
        matches = re.findall(pattern, scrubbed, re.IGNORECASE)
        removed_dates.extend(matches)
        scrubbed = re.sub(pattern, '', scrubbed, flags=re.IGNORECASE)
    
    # Pattern 2: Standalone years (4 digits that look like years)
    # Be careful not to remove numbers that aren't years
    year_pattern = r'\b(19[89]\d|20[0-2]\d)\b'
    
    def replace_year(match):
        year = match.group(1)
        # Check context - only remove if it looks like a date context
        start = max(0, match.start() - 20)
        end = min(len(scrubbed), match.end() + 20)
        context = scrubbed[start:end].lower()
        
        date_contexts = ['in', 'year', 'since', 'from', 'during', 'by', 'until', 
                        'january', 'february', 'march', 'april', 'may', 'june',
                        'july', 'august', 'september', 'october', 'november', 'december',
                        'dated', 'published', 'written']
        
        if any(ctx in context for ctx in date_contexts):
            removed_dates.append(year)
            return ''
        return year  # Keep it if not in date context
    
    scrubbed = re.sub(year_pattern, replace_year, scrubbed)
    
    # Pattern 3: Relative time expressions with specific timeframes
    relative_patterns = [
        r'\blast\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
        r'\bthis\s+(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
        r'\b(?:on|last|this)\s+\d{1,2}(?:st|nd|rd|th)?\b',
    ]
    
    for pattern in relative_patterns:
        matches = re.findall(pattern, scrubbed, re.IGNORECASE)
        removed_dates.extend(matches)
        scrubbed = re.sub(pattern, '', scrubbed, flags=re.IGNORECASE)
    
    # Pattern 4: ISO dates like 2017-01-15
    iso_pattern = r'\b\d{4}-\d{2}-\d{2}\b'
    matches = re.findall(iso_pattern, scrubbed)
    removed_dates.extend(matches)
    scrubbed = re.sub(iso_pattern, '', scrubbed)
    
    # Pattern 5: Slash dates like 01/15/2017 or 15/01/2017
    slash_pattern = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b'
    matches = re.findall(slash_pattern, scrubbed)
    removed_dates.extend(matches)
    scrubbed = re.sub(slash_pattern, '', scrubbed)
    
    # Clean up extra whitespace
    scrubbed = re.sub(r'\s+', ' ', scrubbed)
    scrubbed = re.sub(r'\s+([.,;:!?])', r'\1', scrubbed)
    scrubbed = scrubbed.strip()
    
    return scrubbed, removed_dates


def process_article(text: str, year: int) -> ProcessedArticle:
    """
    Process a single article: scrub dates and prepare for use.
    
    Args:
        text: Original article text
        year: Publication year
        
    Returns:
        ProcessedArticle with scrubbed text
    """
    scrubbed_text, dates_removed = scrub_explicit_dates(text)
    
    return ProcessedArticle(
        original_text=text,
        scrubbed_text=scrubbed_text,
        year=year,
        dates_removed=dates_removed
    )


def extract_year_from_metadata(metadata: dict) -> Optional[int]:
    """
    Extract publication year from article metadata.
    Handles various metadata formats.
    
    Args:
        metadata: Article metadata dictionary
        
    Returns:
        Year as integer, or None if not found
    """
    # Common keys for date/year in metadata
    date_keys = ['date', 'published', 'publish_date', 'publication_date', 
                 'created', 'timestamp', 'year']
    
    for key in date_keys:
        if key in metadata:
            value = metadata[key]
            
            # If it's already an integer year
            if isinstance(value, int) and 1900 <= value <= 2030:
                return value
            
            # If it's a string, try to extract year
            if isinstance(value, str):
                # Try to find a 4-digit year
                match = re.search(r'(19|20)\d{2}', value)
                if match:
                    return int(match.group())
    
    return None


# Test the scrubbing
if __name__ == "__main__":
    test_article = """
    Published on January 15, 2017
    
    The economy continues to show signs of recovery as the Obama administration 
    pushes for new infrastructure investments in 2016. The report dated 2017-01-10 
    shows that since 2015, unemployment has dropped significantly.
    
    Meanwhile, users are flocking to Vine to share short video clips, and everyone 
    is talking about the upcoming Star Wars sequel. Last Tuesday, the stock market 
    reached new highs.
    """
    
    result = process_article(test_article, 2017)
    
    print("Original:")
    print(result.original_text)
    print("\nScrubbed:")
    print(result.scrubbed_text)
    print("\nDates removed:")
    print(result.dates_removed)

