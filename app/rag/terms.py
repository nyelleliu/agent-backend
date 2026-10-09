import re


LATIN_TOKEN = re.compile(r"[a-zA-Z0-9_]{2,}")
CJK_RUN = re.compile(r"[一-鿿]+")

LATIN_STOPWORDS = {
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "all",
    "any",
    "can",
    "her",
    "was",
    "one",
    "our",
    "out",
    "has",
    "have",
    "this",
    "that",
    "with",
    "from",
    "what",
    "when",
    "where",
    "who",
    "how",
    "why",
    "does",
    "did",
}

CJK_STOP_CHARS = set(
    "的了是在我你他她它们有和也不吗呢啊吧及或被把给对从向就都还很"
)


def extract_terms(text: str, limit: int | None = None) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str):
        if term in seen:
            return

        seen.add(term)
        terms.append(term)

    for token in LATIN_TOKEN.findall((text or "").lower()):
        if token in LATIN_STOPWORDS:
            continue

        add(token)

    for run in CJK_RUN.findall(text or ""):
        for index in range(len(run) - 1):
            bigram = run[index:index + 2]

            if (
                bigram[0] in CJK_STOP_CHARS
                and bigram[1] in CJK_STOP_CHARS
            ):
                continue

            add(bigram)

    if limit is not None:
        return terms[:limit]

    return terms
