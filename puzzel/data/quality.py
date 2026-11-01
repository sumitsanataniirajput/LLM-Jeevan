"""Unicode-preserving text normalization, quality checks, near-duplicate detection, and content categorization."""

from __future__ import annotations

import io
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from itertools import islice

_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")


def normalize_text(text: str) -> str:
    """Normalize Unicode/spacing while preserving punctuation and paragraphs."""
    if not isinstance(text, str):
        raise TypeError("document text must be a string")
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    output = io.StringIO()
    has_content = False
    pending_blank = False
    for raw_line in io.StringIO(text):
        line = _HORIZONTAL_WHITESPACE.sub(" ", raw_line.rstrip("\n")).strip()
        if not line:
            if has_content:
                pending_blank = True
            continue
        if has_content:
            output.write("\n\n" if pending_blank else "\n")
        output.write(line)
        has_content = True
        pending_blank = False
    return output.getvalue()


def quality_rejection_reason(text: str) -> str | None:
    """Return a rejection reason for clearly corrupted/repetitive documents.

    Heuristics are intentionally conservative for multilingual material. They
    sample at most 10,000 characters/windows, so checking a bounded document
    does not allocate memory proportional to its full character count.
    """
    if "\ufffd" in text:
        return "replacement_character"
    if "\x00" in text:
        return "nul_character"

    if text:
        control_count = sum(
            unicodedata.category(character) == "Cc"
            and character not in "\n\r\t\f"
            for character in text
        )
        if control_count >= 2 and control_count / len(text) > 0.01:
            return "excessive_control_characters"

    if len(text) < 100:
        return None

    sample_stride = max(1, len(text) // 10_000)
    sample = text[::sample_stride][:10_000]
    char_counts = Counter(sample)
    if char_counts and max(char_counts.values()) / len(sample) >= 0.92:
        return "excessive_single_character_repetition"

    sampled_lines = [
        line.strip()
        for line in islice(io.StringIO(text), 2048)
        if line.strip()
    ]
    if len(sampled_lines) >= 10:
        line_counts = Counter(sampled_lines)
        if max(line_counts.values()) / len(sampled_lines) >= 0.9:
            return "excessive_repeated_lines"

    if len(sample) >= 100:
        ngram_counts = Counter(
            sample[index : index + 5] for index in range(len(sample) - 4)
        )
        ngram_total = sum(ngram_counts.values())
        if ngram_total and len(ngram_counts) / ngram_total < 0.05:
            return "excessive_repeated_ngrams"
    return None


# ---------------------------------------------------------------------------
# Near-duplicate detection via MinHash signatures
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class NearDuplicateResult:
    """Result of near-duplicate comparison between two documents."""

    is_near_duplicate: bool
    jaccard_estimate: float
    shingle_size: int
    num_perm: int


def _shingles(text: str, size: int = 5) -> set[str]:
    """Return the set of character-level shingles of the given size."""
    if len(text) < size:
        return {text}
    return {text[index : index + size] for index in range(len(text) - size + 1)}


def _minhash_signature(shingles: set[str], num_perm: int = 128) -> list[int]:
    """Compute a MinHash signature for a set of shingles.

    Uses a family of simple hash functions parameterized by prime modulus.
    Deterministic and dependency-free.
    """
    if not shingles:
        return [0] * num_perm

    # Large prime for modular hashing.
    prime = (1 << 61) - 1  # 2^61 - 1, a Mersenne prime
    # Random but fixed coefficients for each permutation.
    coefficients = [
        (i * 2654435761 + 1, i * 40503 + 7) for i in range(num_perm)
    ]

    signature = [prime] * num_perm
    for shingle in shingles:
        shingle_hash = _hash_string(shingle) % prime
        for i, (a, b) in enumerate(coefficients):
            h = (a * shingle_hash + b) % prime
            if h < signature[i]:
                signature[i] = h
    return signature


def _hash_string(text: str) -> int:
    """Deterministic 64-bit hash of a string."""
    h = 1469598103934665603  # FNV offset basis
    for char in text.encode("utf-8"):
        h = ((h * 1099511628211) & ((1 << 64) - 1)) ^ char
    return h


def estimate_jaccard(sig1: list[int], sig2: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("signatures must have the same length")
    if not sig1:
        return 0.0
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def is_near_duplicate(
    text1: str,
    text2: str,
    *,
    shingle_size: int = 5,
    num_perm: int = 128,
    threshold: float = 0.85,
) -> NearDuplicateResult:
    """Determine whether two documents are near-duplicates.

    Uses character-level MinHash with Jaccard similarity estimation.
    Documents above the similarity threshold are considered near-duplicates.

    Args:
        text1: First document text.
        text2: Second document text.
        shingle_size: Character n-gram size for shingling.
        num_perm: Number of MinHash permutations.
        threshold: Jaccard similarity threshold for near-duplicate classification.

    Returns:
        NearDuplicateResult with the classification and estimated similarity.
    """
    sig1 = _minhash_signature(_shingles(text1, shingle_size), num_perm)
    sig2 = _minhash_signature(_shingles(text2, shingle_size), num_perm)
    jaccard = estimate_jaccard(sig1, sig2)
    return NearDuplicateResult(
        is_near_duplicate=jaccard >= threshold,
        jaccard_estimate=jaccard,
        shingle_size=shingle_size,
        num_perm=num_perm,
    )


# ---------------------------------------------------------------------------
# Content categorization (genre/tags)
# ---------------------------------------------------------------------------

# Keywords that indicate content category. These are language-agnostic
# heuristics based on common words across multiple languages.
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "general_knowledge": [
        "fact", "information", "knowledge", "about", "what is", "who is",
        "hecho", "información", "conocimiento", "qué es", "quién es",
        "fait", "information", "connaissance", "qu'est-ce que", "qui est",
        "حقيقة", "معلومات", "معرفة", "ما هو", "من هو",
        "事实", "信息", "知识", "是什么", "谁是",
    ],
    "educational": [
        "lesson", "study", "learn", "teach", "school", "education",
        "lección", "estudio", "aprender", "enseñar", "escuela", "educación",
        "leçon", "étude", "apprendre", "enseigner", "école", "éducation",
        "درس", "دراسة", "تعلم", "يعلّم", "مدرسة", "教育",
        "课程", "学习", "教学", "学校", "教育",
    ],
    "science": [
        "science", "scientific", "experiment", "research", "theory",
        "ciencia", "científico", "experimento", "investigación", "teoría",
        "science", "scientifique", "expérience", "recherche", "théorie",
        "علم", "علمي", "تجربة", "بحث", "نظرية",
        "科学", "科学的", "实验", "研究", "理论",
    ],
    "technology": [
        "technology", "technical", "computer", "software", "engineering",
        "tecnología", "técnico", "computadora", "software", "ingeniería",
        "technologie", "technique", "ordinateur", "logiciel", "ingénierie",
        "تكنولوجيا", "تقني", "كمبيوتر", "برمجيات", "هندسة",
        "技术", "技术的", "计算机", "软件", "工程",
    ],
    "history": [
        "history", "historical", "century", "war", "ancient", "empire",
        "historia", "histórico", "siglo", "guerra", "antiguo", "imperio",
        "histoire", "historique", "siècle", "guerre", "ancien", "empire",
        "تاريخ", "تاريخي", "قرن", "حرب", "قديم", "إمبراطورية",
        "历史", "历史的", "世纪", "战争", "古代", "帝国",
    ],
    "geography": [
        "geography", "geographic", "country", "city", "mountain", "river",
        "geografía", "geográfico", "país", "ciudad", "montaña", "río",
        "géographie", "géographique", "pays", "ville", "montagne", "rivière",
        "جغرافيا", "جغرافي", "بلد", "مدينة", "جبل", "نهر",
        "地理", "地理的", "国家", "城市", "山", "河",
    ],
    "literature": [
        "novel", "story", "poem", "poetry", "fiction", "literature",
        "novela", "historia", "poema", "poesía", "ficción", "literatura",
        "roman", "histoire", "poème", "poésie", "fiction", "littérature",
        "رواية", "قصة", "قصيدة", "شعر", "خيال", "أدب",
        "小说", "故事", "诗", "诗歌", "虚构", "文学",
    ],
    "mathematics": [
        "mathematics", "math", "algebra", "geometry", "equation", "number",
        "matemáticas", "matemática", "álgebra", "geometría", "ecuación", "número",
        "mathématiques", "maths", "algèbre", "géométrie", "équation", "nombre",
        "رياضيات", "حاسب", "جبر", "هندسة", "معادلة", "عدد",
        "数学", "代数", "几何", "方程", "数字",
    ],
    "programming": [
        "programming", "code", "program", "software", "algorithm", "function",
        "programación", "código", "programa", "software", "algoritmo", "función",
        "programmation", "code", "programme", "logiciel", "algorithme", "fonction",
        "برمجة", "كود", "برنامج", "برمجيات", "خوارزمية", "دالة",
        "编程", "代码", "程序", "软件", "算法", "函数",
    ],
    "conversation": [
        "hello", "how are you", "goodbye", "please", "thank you", "yes", "no",
        "hola", "cómo estás", "adiós", "por favor", "gracias", "sí", "no",
        "bonjour", "comment allez-vous", "au revoir", "s'il vous plaît", "merci", "oui", "non",
        "مرحبا", "كيف حالك", "مع السلامة", "من فضلك", "شكراً", "نعم", "لا",
        "你好", "你好吗", "再见", "请", "谢谢", "是", "不是",
    ],
    "creative_writing": [
        "once upon a time", "story", "tale", "character", "imagine",
        "érase una vez", "cuento", "historia", "personaje", "imagina",
        "il était une fois", "conte", "histoire", "personnage", "imagine",
        "كان هناك مرة", "قصة", "حكاية", "شخصية", "تخيل",
        "从前", "故事", "传说", "角色", "想象",
    ],
    "language_learning": [
        "language", "learn", "vocabulary", "grammar", "translation",
        "idioma", "aprender", "vocabulario", "gramática", "traducción",
        "langue", "apprendre", "vocabulaire", "grammaire", "traduction",
        "لغة", "تعلم", "مفردات", "قواعد", "ترجمة",
        "语言", "学习", "词汇", "语法", "翻译",
    ],
}


def categorize_content(text: str) -> list[str]:
    """Categorize document text into content genres.

    Uses keyword heuristics matched against the lowercased text. Returns
    a list of category labels; may be empty if no category is detected.

    Args:
        text: Document text to categorize.

    Returns:
        List of matching category labels.
    """
    lowered = text.lower()
    matched: list[str] = []
    for category, keywords in _CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lowered:
                matched.append(category)
                break
    return matched


def detect_language_from_text(text: str) -> str | None:
    """Detect the language of text using Unicode script analysis.

    Wrapper around :func:`puzzel.languages.detect_language` for use
    during dataset ingestion.

    Args:
        text: Text to analyze.

    Returns:
        Detected language code, or None if undetectable.
    """
    from puzzel.languages import detect_language

    return detect_language(text)
