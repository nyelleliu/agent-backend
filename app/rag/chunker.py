import re
from dataclasses import dataclass


PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
SENTENCE_SPLIT = re.compile(
    r"(?<=[。！？!?；;…])|(?<=[.!?])\s+|\n+"
)


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


class TextChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 80):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.chunk_size = chunk_size
        self.overlap = max(0, min(overlap, chunk_size // 4))

    def split(self, text: str) -> list[Chunk]:
        normalized = (text or "").replace("\r\n", "\n").strip()

        if not normalized:
            return []

        units = self._split_units(normalized)
        packed = self._pack(units)

        return [
            Chunk(index=index, text=value)
            for index, value in enumerate(packed)
        ]

    def _split_units(self, text: str) -> list[tuple[str, str]]:
        units: list[tuple[str, str]] = []

        for paragraph in PARAGRAPH_SPLIT.split(text):
            paragraph = paragraph.strip()

            if not paragraph:
                continue

            separator = "\n" if units else ""

            if len(paragraph) <= self.chunk_size:
                units.append((paragraph, separator))
                continue

            for sentence in SENTENCE_SPLIT.split(paragraph):
                sentence = sentence.strip()

                if not sentence:
                    continue

                units.append((sentence, separator))
                separator = ""

        broken: list[tuple[str, str]] = []

        for value, separator in units:
            for index, piece in enumerate(self._break(value)):
                broken.append((piece, separator if index == 0 else ""))

        return broken

    def _break(self, unit: str) -> list[str]:
        if len(unit) <= self.chunk_size:
            return [unit]

        step = self.chunk_size - self.overlap

        if step <= 0:
            step = self.chunk_size

        pieces = []
        start = 0

        while start < len(unit):
            pieces.append(unit[start:start + step])
            start += step

        return pieces

    def _pack(self, units: list[tuple[str, str]]) -> list[str]:
        chunks: list[str] = []
        current = ""

        for value, separator in units:
            if not current:
                current = value
                continue

            if len(current) + len(separator) + len(value) <= self.chunk_size:
                current = current + separator + value
                continue

            chunks.append(current)
            carry = current[-self.overlap:] if self.overlap else ""
            current = (
                carry + value
                if len(carry) + len(value) <= self.chunk_size
                else value
            )

        if current:
            chunks.append(current)

        return chunks
