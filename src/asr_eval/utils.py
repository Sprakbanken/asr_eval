import re


def remove_punctuation(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def remove_underscore(text: str) -> str:
    text = re.sub(r"_", " ", text)
    return text


def remove_multiple_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def remove_hesitations(text: str) -> str:
    text = re.sub(r"(?<!\w)eee(?!\w)", "", text)
    text = re.sub(r"(?<!\w)mmm(?!\w)", "", text)
    text = re.sub(r"(?<!\w)qqq(?!\w)", "", text)
    text = re.sub(r"(?<!\w)eh(?!\w)", "", text)
    text = re.sub(r"(?<!\w)ehm(?!\w)", "", text)
    text = re.sub(r"(?<!\w)mhm(?!\w)", "", text)
    text = re.sub(r"(?<!\w)mm(?!\w)", "", text)
    text = re.sub(r"(?<!\w)hmmm(?!\w)", "", text)
    text = re.sub(r"(?<!\w)hmm(?!\w)", "", text)
    text = re.sub(r"(?<!\w)hm(?!\w)", "", text)
    return text


def standardize_text(text: str) -> str:
    text = text.lower()
    text = remove_hesitations(text)
    text = remove_punctuation(text)
    text = remove_multiple_spaces(text)
    return text
