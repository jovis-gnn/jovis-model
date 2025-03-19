STRING_CTGRS = [
    "BAGS",
    "HAIR_ACCESSORIES",
    "HATS",
    "KEY_RING",
    "OUTWEARS",
    "PANTS",
    "SCARF/MUFFLER",
    "SKIRTS",
    "SOCKS",
    "SWIMWEARS",
    "TIE",
    "TOPS",
    "WHOLEBODIES",
]


UPPERCASE_LATIN = list(range(int("0x0041", base=16), int("0x005A", base=16)))
LOWERCASE_LATIN = list(range(int("0x0061", base=16), int("0x007A", base=16)))
HANGUL_JAMO = list(range(int("0x1100", base=16), int("0x11FF", base=16)))
HANGUL_SYLLABLES = list(range(int("0xAC00", base=16), int("0xD7AF", base=16)))
ENGLISH = UPPERCASE_LATIN + LOWERCASE_LATIN
KOREAN = HANGUL_JAMO + HANGUL_SYLLABLES


SPACE_CHARS = [" ", "~", "_", "-", "(", ")", "{", "}", "[", "]", "/", ",", ".", "|"]
