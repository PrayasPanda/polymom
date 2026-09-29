"""Flags utterances that look like instructions aimed at the model (prompt injection).

Detection only: flagged utterances are still summarized as ordinary speech and
listed in ``verification_report.injection_flags``. The real defences are the
delimited data blocks, the system prompt and schema-validated output.
"""

import re
import unicodedata
from collections.abc import Sequence

from app.schemas.summary import InjectionFlag
from app.schemas.transcript import Utterance

PATTERNS: dict[str, re.Pattern[str]] = {
    "en_ignore_instructions": re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|your)"
        r"\b.{0,20}\b(instructions?|prompts?|rules)\b"
    ),
    "en_system_prompt": re.compile(r"\b(system|developer|hidden)\s+(prompt|message|instructions?)"),
    "en_role_change": re.compile(r"\b(you are now|act as|pretend to be|jailbreak|dan mode)\b"),
    "en_output_control": re.compile(
        r"\b(respond|reply|output|return)\b.{0,20}\b(only|exactly)\b.{0,30}\b(json|text|with)\b"
    ),
    "hi_ignore_instructions": re.compile(
        r"(पिछले|पुराने|सभी|ऊपर के)?\s*(निर्देश|निर्देशों|आदेश)\S*\s*(को\s*)?"
        r"(अनदेखा|नज़रअंदाज़|नजरअंदाज|भूल)"
    ),
    "hi_system_prompt": re.compile(r"सिस्टम\s*(प्रॉम्प्ट|प्रांप्ट|संदेश)"),
    "hi_romanized": re.compile(
        r"\b(pichhle|pichle|saare|sabhi)\s+(instructions?|nirdesh)\b.{0,20}\b(ignore|bhool|bhul)"
        r"|\b(instructions?|nirdesh)\s+(ko\s+)?(ignore|bhool|bhul)\s*(karo|kar do|jao)"
    ),
    "or_ignore_instructions": re.compile(r"(ପୂର୍ବ|ପୂର୍ବର|ସମସ୍ତ)?\s*ନିର୍ଦ୍ଦେଶ\S*\s*(କୁ\s*)?(ଅଣଦେଖା|ଉପେକ୍ଷା|ଭୁଲି)"),
    "or_system_prompt": re.compile(r"ସିଷ୍ଟମ\s*(ପ୍ରମ୍ପ୍ଟ|ପ୍ରମ୍ପଟ|ବାର୍ତ୍ତା)"),
}


def detect_injections(utterances: Sequence[Utterance]) -> list[InjectionFlag]:
    flags: list[InjectionFlag] = []
    for u in utterances:
        text = unicodedata.normalize("NFC", u.text).casefold()
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                flags.append(
                    InjectionFlag(utterance_id=u.id, speaker=u.speaker, pattern=name, text=u.text)
                )
                break
    return flags
