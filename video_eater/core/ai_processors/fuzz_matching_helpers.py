import logging
import re
from copy import deepcopy
from typing import Tuple, Optional, List

from rapidfuzz import fuzz, process

from video_eater.core.ai_processors.ai_prompt_models import PullQuoteWithTimestamp, PullQuote

logger = logging.getLogger(__name__)


def extract_timestamp_from_srt_position(srt_text: str, position: int) -> float:
    """Extract timestamp from SRT text at given character position."""
    # Find the last timestamp before this position
    timestamp_pattern = r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"

    # Get all timestamps before this position
    text_before = srt_text[:position]
    timestamps = list(re.finditer(timestamp_pattern, text_before))

    if timestamps:
        last_timestamp = timestamps[-1]
        h, m, s, ms = last_timestamp.groups()
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    logger.warning(f"No timestamp found before position {position} for quote: {text_before[-30:]} - returning 0.0")
    return 0.0


def find_best_match_in_transcript(
        quote_text: str,
        transcript_srt: str,
        min_similarity: float = 75.0
) -> Optional[Tuple[int, int, float]]:
    """
    Find the best fuzzy match for a quote in the transcript using RapidFuzz.

    Returns:
        Optional tuple of (start_position, end_position, similarity_score)
    """
    # Normalize the quote for matching
    normalized_quote = normalize_text(quote_text)
    quote_length = len(quote_text)  # Use original length for window sizing

    if len(normalized_quote) < 10:
        logger.warning(f"Quote too short for reliable matching: {quote_text}")
        return None

    # Create sliding windows for candidate matching
    window_size = min(quote_length * 2, 500)  # Cap window size for performance
    step_size = max(20, quote_length // 10)

    candidates = []
    positions = []

    for i in range(0, len(transcript_srt) - window_size + 1, step_size):
        window = transcript_srt[i:i + window_size]
        candidates.append(window)
        positions.append(i)

    if not candidates:
        return None

    # Use RapidFuzz's partial_ratio for finding quotes within larger text
    best_match = None
    best_score = 0

    # First pass: Try partial_ratio for flexible matching
    for window, position in zip(candidates, positions):
        score = fuzz.partial_ratio(quote_text, window)

        if score > best_score and score >= min_similarity:
            best_score = score
            # Try to find more precise position within the window
            precise_pos = find_precise_position(quote_text, window, position)
            if precise_pos:
                start_pos, end_pos = precise_pos
                best_match = (start_pos, end_pos, score / 100.0)

    # If no good match found, try token-based matching for speech
    if not best_match or best_score < min_similarity:
        best_match = find_token_based_match(
            quote_text,
            transcript_srt,
            candidates,
            positions,
            min_similarity
        )

    # If still no match, try aggressive fuzzy matching
    if not best_match:
        best_match = find_fuzzy_match_aggressive(
            quote_text,
            transcript_srt,
            min_similarity
        )

    return best_match


def find_precise_position(
        quote: str,
        window: str,
        window_start: int
) -> Optional[Tuple[int, int]]:
    """
    Find the precise position of a quote within a window.
    Uses token alignment for better accuracy.
    """
    # Try exact substring first
    quote_lower = quote.lower()
    window_lower = window.lower()

    idx = window_lower.find(quote_lower)
    if idx != -1:
        return (window_start + idx, window_start + idx + len(quote))

    # Use RapidFuzz's optimal string alignment to find best position
    best_score = 0
    best_pos = None

    # Slide through window with quote-sized chunks
    for i in range(len(window) - len(quote) + 1):
        chunk = window[i:i + len(quote)]
        score = fuzz.ratio(quote, chunk)

        if score > best_score:
            best_score = score
            best_pos = (window_start + i, window_start + i + len(quote))

    return best_pos if best_score > 70 else None


def find_token_based_match(
        quote_text: str,
        transcript_srt: str,
        candidates: List[str],
        positions: List[int],
        min_similarity: float
) -> Optional[Tuple[int, int, float]]:
    """
    Use token-based matching which handles word order variations and fillers better.
    """
    best_match = None
    best_score = 0

    for window, position in zip(candidates, positions):
        # Try different token-based scorers
        scores = [
            fuzz.token_set_ratio(quote_text, window),  # Ignores duplicates
            fuzz.token_sort_ratio(quote_text, window),  # Ignores word order
            fuzz.WRatio(quote_text, window)  # Weighted combination
        ]

        max_score = max(scores)

        if max_score > best_score and max_score >= min_similarity:
            best_score = max_score
            # Find approximate position
            start_pos = position
            end_pos = min(position + len(quote_text), len(transcript_srt))
            best_match = (start_pos, end_pos, max_score / 100.0)

    return best_match


def find_fuzzy_match_aggressive(
        quote_text: str,
        transcript_srt: str,
        min_similarity: float = 70.0
) -> Optional[Tuple[int, int, float]]:
    """
    More aggressive fuzzy matching using RapidFuzz's process.extract.
    """
    # Try different strategies to extract matchable portions
    strategies = [
        # Strategy 1: Middle 60% of quote
        lambda q: q[len(q) // 5:-len(q) // 5] if len(q) > 20 else q,

        # Strategy 2: First 30 words
        lambda q: ' '.join(q.split()[:30]),

        # Strategy 3: Last 30 words
        lambda q: ' '.join(q.split()[-30:]),

        # Strategy 4: Remove fillers and try again
        lambda q: remove_fillers(q),

        # Strategy 5: Key content words only
        lambda q: extract_key_words(q)
    ]

    for strategy in strategies:
        modified_quote = strategy(quote_text)
        if len(modified_quote) < 10:
            continue

        # Create smaller windows for the modified quote
        window_size = min(len(modified_quote) * 3, 300)
        candidates = []
        positions = []

        for i in range(0, len(transcript_srt) - window_size + 1, 50):
            window = transcript_srt[i:i + window_size]
            candidates.append(window)
            positions.append(i)

        if not candidates:
            continue

        # Use process.extractOne for best match
        result = process.extractOne(
            modified_quote,
            candidates,
            scorer=fuzz.partial_ratio,
            score_cutoff=min_similarity
        )

        if result:
            matched_text, score, idx = result
            position = positions[idx]

            # Try to expand to full quote boundaries
            start_pos, end_pos = expand_match_boundaries(
                transcript_srt,
                position,
                position + len(modified_quote),
                quote_text
            )

            return (start_pos, end_pos, score / 100.0)

    return None


def normalize_text(text: str) -> str:
    """Normalize text for matching by removing extra whitespace and punctuation."""
    # Remove timestamps if present
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}', '', text)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)  # Remove subtitle numbers

    # Normalize whitespace and case
    text = ' '.join(text.lower().split())

    # Remove or normalize punctuation for matching
    text = re.sub(r'[,;:]', ' ', text)  # Replace some punctuation with spaces
    text = re.sub(r'[.!?]', '', text)  # Remove sentence-ending punctuation
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace again

    return text.strip()


def remove_fillers(text: str) -> str:
    """Remove common speech fillers and hesitations."""
    fillers = [
        r'\b(um+|uh+|ah+|oh+|hmm+|mmm+|err+)\b',
        r'\b(you know|i mean|like|so|well|actually|basically|literally)\b',
        r'\.{2,}',  # Multiple dots
        r'-+',  # Dashes indicating pauses
    ]

    result = text
    for filler in fillers:
        result = re.sub(filler, ' ', result, flags=re.IGNORECASE)

    return ' '.join(result.split())


def extract_key_words(text: str) -> str:
    """Extract key content words, removing common words."""
    common_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'could',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }

    words = text.lower().split()
    key_words = [w for w in words if w not in common_words and len(w) > 2]
    return ' '.join(key_words[:20])  # Limit to first 20 key words


def expand_match_boundaries(
        transcript: str,
        start: int,
        end: int,
        target_quote: str
) -> Tuple[int, int]:
    """
    Expand match boundaries to try to capture the full quote.
    Uses RapidFuzz for similarity comparison.
    """
    # Look for sentence boundaries
    sentence_start = start
    sentence_end = end

    # Expand backward to find sentence start
    for i in range(start - 1, max(0, start - 200), -1):
        if transcript[i] in '.!?\n':
            sentence_start = i + 1
            break

    # Expand forward to find sentence end
    for i in range(end, min(len(transcript), end + 200)):
        if transcript[i] in '.!?\n':
            sentence_end = i
            break

    # Check if expanded match is more similar to target using RapidFuzz
    expanded_text = transcript[sentence_start:sentence_end]
    original_text = transcript[start:end]

    expanded_sim = fuzz.ratio(target_quote, expanded_text) / 100.0
    original_sim = fuzz.ratio(target_quote, original_text) / 100.0

    if expanded_sim > original_sim:
        return sentence_start, sentence_end
    return start, end


async def match_quotes_to_transcript_srt(
        quotes: list[PullQuote],
        transcript_srt: str,
        start_time_offset: float = 0.0
) -> list:
    """
    Match a list of pull quotes to their timestamps in the transcript using RapidFuzz.

    Args:
        quotes: List of quote dictionaries with 'quote_text' or 'text_content' field
        transcript_srt: SRT-formatted transcript text
        start_time_offset: Offset to add to all extracted timestamps (in seconds)

    Returns:
        List of PullQuoteWithTimestamp objects
    """

    quotes_with_timestamps: List[PullQuoteWithTimestamp] = []


    # Match each quote
    for quote in quotes:
        # Try to find the quote in the transcript
        match_result = find_best_match_in_transcript(
            quote.text_content,
            transcript_srt,
            min_similarity=70.0  # Lower threshold for better recall with RapidFuzz
        )

        if match_result:
            start_pos, end_pos, similarity = match_result
            timestamp = extract_timestamp_from_srt_position(transcript_srt, start_pos)

            logger.info(
                f"Matched quote with {similarity:.1%} similarity: "
                f"'{quote.text_content[:50]}...' at {timestamp:.1f}s (offset {start_time_offset:.1f}s)"
            )

            quotes_with_timestamps.append(PullQuoteWithTimestamp(**quote.model_dump(), timestamp_seconds=timestamp + start_time_offset))
        else:
            logger.warning(
                f"Could not find timestamp for quote: '{quote.text_content[:50]}...'"
            )
            # Add with default timestamp at 0.0
            quotes_with_timestamps.append(
                PullQuoteWithTimestamp(**quote.model_dump(), timestamp_seconds=0.0 + start_time_offset))
    return quotes_with_timestamps


def batch_match_quotes(
        quotes: List[str],
        transcript: str,
        min_similarity: float = 75.0
) -> List[Optional[Tuple[int, int, float]]]:
    """
    Batch process multiple quotes for better performance.
    RapidFuzz can process multiple queries efficiently.
    """
    # Create candidate windows from transcript
    window_size = 200
    step = 50

    candidates = []
    positions = []

    for i in range(0, len(transcript) - window_size + 1, step):
        window = transcript[i:i + window_size]
        candidates.append(window)
        positions.append(i)

    results = []

    for quote in quotes:
        # Use process.extractOne for each quote
        match = process.extractOne(
            quote,
            candidates,
            scorer=fuzz.partial_ratio,
            score_cutoff=min_similarity
        )

        if match:
            matched_text, score, idx = match
            position = positions[idx]
            results.append((position, position + len(quote), score / 100.0))
        else:
            results.append(None)

    return results