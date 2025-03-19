import re

from models.sasrec.string_matcher.config import ENGLISH, KOREAN, SPACE_CHARS


def is_korean(text: str) -> bool:
    """Uses Unicode to determine if _any_ chars are Kor."""
    for char in text:
        if ord(char) in KOREAN:
            return True

    return False


def same_lang(text: str, idx1: int, idx2: int) -> bool:
    """
    Determines whether or not two chars are the same language.

    Args:
        text: Input text containing chars.
        idx1: Idx of char1.
        idx2: Idx of char2.

    Example:
        text: "I went to 서울 last year."
        idx1: 8 ('o')
        idx2: 10 ('서')

        same_lang(text, idx1, idx2) == False
    """
    if (not is_korean(text[idx1])) and (not is_korean(text[idx2])):
        return True

    if is_korean(text[idx1]) and is_korean(text[idx2]):
        return True

    return False


def is_start_of_word(text: str, start: int) -> bool:
    """
    Check if the subword is at the start of the word.
        ex) `티` in `티셔츠`.
    """
    if start == 0:
        return True

    if text[start - 1] in SPACE_CHARS:
        return True

    return False


def is_end_of_word(text: str, end: int) -> bool:
    """
    Check if the subword is at the end of the word.
        ex) `ring` in `earring`.
    """
    if end == len(text):
        return True

    if not same_lang(text, end - 1, end):
        return True

    if text[end] in SPACE_CHARS:
        return True

    return False


def is_subword(text: str, span: list[int, int]) -> bool:
    """
    Determines whether the given span is contained by another word\
        within text.

    Case 1: Start and end are at neither end of text.
        Condition 1: Char before entity is not a space character.
        Condition 2: Char after entity is not a space character.

    Case 2: Entity is at the end of the text.
        Condition 1: Char before entity is not a space character.

    Case 3: Entity is at the start of the text:
        Condition 1: Char after entity is not a space character.
    """
    start, end = span

    if (start != 0) and (end != len(text)):
        return (text[start - 1] not in SPACE_CHARS) or (text[end] not in SPACE_CHARS)

    if (start != 0) and (end == len(text)):
        return text[start - 1] not in SPACE_CHARS

    if (start == 0) and (end != len(text)):
        return text[end] not in SPACE_CHARS

    return False


def is_singular_form(text: str, end: int) -> bool:
    """
    Function to check if a word is a singular subword of \
        its plural form.

    ex) `sweater`, `sweaters`
    """
    if end == len(text):
        return False

    if text[end] == "s":
        if end + 1 == len(text):
            return True

        if (ord(text[end + 1]) not in ENGLISH) or (ord(text[end + 1]) not in KOREAN):
            return True

    return False


def remove_black_watch(text: str) -> str:
    """
    Removes all instances of things related to the Black Watch.
    Removing because problematic with actual watches.
    """
    black_watch = ["black watch", "블랙워치", "블랙와치", "블랙 워치", "블랙 와치"]
    for bw in black_watch:
        if bw in text:
            text = text.replace(bw, "")

    return text


def remove_brackets_and_contents(text: str) -> str:
    """
    Content inside parentheses are often redundant and distracting.
    Removing from product names.
    """
    pattern = r"\[.*?\]|\(.*?\)|\{.*?\}|\<.*?\>"
    text = re.sub(pattern=pattern, repl="", string=text).strip()
    return text


def remove_tie_dye(text: str) -> str:
    """
    "Tie-dye" should not be matched with "tie."
    """
    tiedye = ["tie dye", "tie-dye", "타이다이", "타이 다이"]
    for td in tiedye:
        if td in text:
            text = text.replace(td, "")

    return text


def find_latest_entity(label2starts: dict[str, set[int]]) -> str:
    """Finds the matched entity appearing latest in the text."""
    curr_max = -1
    latest_label = ""
    for label, starts in label2starts.items():
        for start in starts:
            if start > curr_max:
                latest_label = label
                curr_max = start

    return latest_label, curr_max
