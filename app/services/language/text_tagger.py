"""Word-level script and language tagging for code-mixed transcripts.

* **Script** comes from Unicode blocks: Devanagari (U+0900-U+097F) -> ``Deva``,
  Odia (U+0B00-U+0B7F) -> ``Orya``, Latin letters -> ``Latn``.
* **Language:** Devanagari words are Hindi, Odia-script words are Odia.
* **Romanized Indic:** Latin-script words found in a small lexicon of frequent,
  unambiguous romanized Hindi / Odia function and filler words are tagged
  ``hi-Latn`` / ``or-Latn`` (e.g. "kal 5 baje hai"). Everything else in Latin
  script is English. The lexicons deliberately leave out words that are also
  common English words ("main", "the", "to", "pain", ...) - precision over
  recall, no heavy model.
* Numbers and punctuation are language-neutral (``None``).
"""

import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from app.schemas.asr import TranscriptSegment, Word

# Frequent romanized Hindi words that are not also common English words.
ROMAN_HINDI: frozenset[str] = frozenset(
    """
    hai hain tha thi hoga hogi honge kya kyun kyon kaise kaisa kab kahan kaun
    nahi nahin mat aur lekin magar ya phir abhi kal parso aaj kyunki isliye
    mein mujhe mera meri mere hum humein hamara aap aapka aapki tum tumhara
    yeh woh wahi yahi kuch sab sabhi bahut bohot zyada kam thoda accha acha
    achha theek thik haan ji karo karna karenge karega karegi kiya kiye raha
    rahe rahi gaya gayi gaye chahiye sakte sakta sakti baje wala wali wale
    matlab chalo dekho bolo samjha samjhe samajh kaam paisa paise log baat
    bataiye batao dijiye kijiye hoon hua hui hue lagta lagti jaldi abhi
    """.split()  # noqa: SIM905 - a word list reads better than a 100-item literal
)
# Frequent romanized Odia words; ambiguous ones ("se", "kaha": also Hindi) are left out.
ROMAN_ODIA: frozenset[str] = frozenset(
    """
    mu mote mora tume tumara apana apananka ame amara tanka kemiti
    kana kahinki kebe kouthi achi achhi achhe nahanti heba hela hebani
    karibi karibe karuchi karuchhi kariba jiba jibi asiba asuchi bhala kichhi
    sabu ebe kalire aji sathire pakhare bhitare upare kahuchi kahibe
    dekhibe bujhila bujhili hau
    """.split()  # noqa: SIM905 - a word list reads better than a 100-item literal
)

_BLOCKS = (("Deva", 0x0900, 0x097F), ("Orya", 0x0B00, 0x0B7F))
LANGUAGE_BY_SCRIPT = {"Deva": "hi", "Orya": "or", "Latn": "en"}


def script_of(word: str) -> str | None:
    """Majority script of the word's letters/marks; ``None`` for digits/punctuation."""
    counts: Counter[str] = Counter()
    for char in word:
        if unicodedata.category(char)[0] not in "LM":
            continue
        code = ord(char)
        for name, lo, hi in _BLOCKS:
            if lo <= code <= hi:
                counts[name] += 1
                break
        else:
            if "LATIN" in unicodedata.name(char, ""):
                counts["Latn"] += 1
            else:
                counts["Other"] += 1
    if not counts:
        return None
    return max(sorted(counts), key=lambda s: counts[s])


def word_language(word: str, script: str | None) -> str | None:
    if script is None or script == "Other":
        return None
    if script == "Latn":
        key = "".join(c for c in word.casefold() if c.isalpha())
        if key in ROMAN_HINDI:
            return "hi-Latn"
        if key in ROMAN_ODIA:
            return "or-Latn"
    return LANGUAGE_BY_SCRIPT[script]


def base_language(tag: str) -> str:
    """``hi-Latn`` -> ``hi``."""
    return tag.split("-", 1)[0]


def tag_word(word: Word) -> Word:
    script = script_of(word.text)
    return word.model_copy(update={"script": script, "language": word_language(word.text, script)})


@dataclass(frozen=True, slots=True)
class CodeMix:
    primary_language: str | None
    languages_present: list[str]
    is_code_mixed: bool
    code_mix_ratio: float


def code_mix_stats(words: Sequence[Word], fallback: str | None = None) -> CodeMix:
    """Segment-level mix from tagged words (base languages: ``hi-Latn`` counts as ``hi``).

    ``primary_language`` is the most frequent base language (ties go to
    ``fallback``, usually the audio-level language); ``code_mix_ratio`` is the
    share of language-tagged words outside it.
    """
    counts = Counter(base_language(w.language) for w in words if w.language)
    if not counts:
        return CodeMix(fallback, [fallback] if fallback else [], False, 0.0)
    top = max(counts.values())
    leaders = sorted(lang for lang, n in counts.items() if n == top)
    primary = fallback if fallback in leaders else leaders[0]
    total = sum(counts.values())
    return CodeMix(
        primary_language=primary,
        languages_present=sorted(counts),
        is_code_mixed=len(counts) > 1,
        code_mix_ratio=round(1 - counts[primary] / total, 4),
    )


def tag_segment(segment: TranscriptSegment) -> TranscriptSegment:
    """Tag words (or the text's tokens when the backend gave no words)."""
    words = segment.words or [
        Word(text=token, start=segment.start, end=segment.end) for token in segment.text.split()
    ]
    tagged = [tag_word(w) for w in words]
    mix = code_mix_stats(tagged, fallback=segment.language)
    return segment.model_copy(
        update={
            "words": tagged if segment.words else [],
            "primary_language": mix.primary_language,
            "languages_present": mix.languages_present,
            "is_code_mixed": mix.is_code_mixed,
            "code_mix_ratio": mix.code_mix_ratio,
        }
    )
