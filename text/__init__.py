import re

def split_and_recombine_text(text, desired_length=100, max_length=180):
    # from https://github.com/neonbjb/tortoise-tts
    """Split text it into chunks of a desired length trying to keep sentences intact."""
    # normalize text, remove redundant whitespace and convert non-ascii quotes to ascii
    text = re.sub(r"\n\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”]", '"', text)

    rv = []
    in_quote = False
    current = ""
    split_pos = []
    pos = -1
    end_pos = len(text) - 1

    def seek(delta):
        nonlocal pos, in_quote, current
        is_neg = delta < 0
        for _ in range(abs(delta)):
            if is_neg:
                pos -= 1
                current = current[:-1]
            else:
                pos += 1
                current += text[pos]
            if text[pos] == '"':
                in_quote = not in_quote
        return text[pos]

    def peek(delta):
        p = pos + delta
        return text[p] if p < end_pos and p >= 0 else ""

    def commit():
        nonlocal rv, current, split_pos
        rv.append(current)
        current = ""
        split_pos = []

    while pos < end_pos:
        c = seek(1)
        # do we need to force a split?
        if len(current) >= max_length:
            if len(split_pos) > 0 and len(current) > (desired_length / 2):
                # we have at least one sentence and we are over half the desired length, seek back to the last split
                d = pos - split_pos[-1]
                seek(-d)
            else:
                # no full sentences, seek back until we are not in the middle of a word and split there
                while c not in "!?.\n " and pos > 0 and len(current) > desired_length:
                    c = seek(-1)
            commit()
        # check for sentence boundaries
        elif not in_quote and (c in "!?\n" or (c == "." and peek(1) in "\n ")):
            # seek forward if we have consecutive boundary markers but still within the max length
            while (
                pos < len(text) - 1 and len(current) < max_length and peek(1) in "!?."
            ):
                c = seek(1)
            split_pos.append(pos)
            if len(current) >= desired_length:
                commit()
        # treat end of quote as a boundary if its followed by a space or newline
        elif in_quote and peek(1) == '"' and peek(2) in "\n ":
            seek(2)
            split_pos.append(pos)
    rv.append(current)

    # clean up, remove lines with only whitespace or punctuation
    rv = [s.strip() for s in rv]
    rv = [s for s in rv if len(s) > 0 and not re.match(r"^[\s\.,;:!?]*$", s)]

    return rv

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
    ('inc', 'incorporated'),
]]

def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text

def lowercase(text):
    return text.lower()

def name_replacer(text):
    text = lowercase(text)
    text = text.replace("ruggiero", "rujero")
    text = text.replace("bufalino", "buffalino")
    text = text.replace("genovese", "gɛnoviz")
    text = text.replace("apalachin", "apalaykin")
    text = text.replace("stefano", "stɛfɛnoʊ")
    text = text.replace("capo", "kapo")
    text = text.replace("deChicco", "de Chicco")
    text = text.replace("giuseppe", "jewseppe")
    text = text.replace("luciano", "lewcheeyano")
    text = text.replace("anc", "a n c")
    text = expand_abbreviations(text)
    return text