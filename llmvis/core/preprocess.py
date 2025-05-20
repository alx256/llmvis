"""
Helper functions for the preprocessing of data for visualizations
"""

BANNED_WORDS = set(
    [
        "is",
        "and",
        "or",
        "a",
        "an",
        "the",
        "for",
        "of",
        "its",
        "it's",
        "are",
        "be",
        "to",
        "in",
        "has",
        "it",
    ]
)


def should_include(word: str) -> bool:
    """
    Used for determining if a given `word` is of interest.
    Useful for visualizations where we are examining different
    words and want to ignore words that are of little interest
    to the user such as connectives ("and", "but" etc.).

    Args:
        word (str): The word that should be checked for if it
            is useful.

    Returns:
        A `bool` that is `true` if this word is of interest and
        `false` if it is not.
    """

    return word not in BANNED_WORDS
