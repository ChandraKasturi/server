import re
from typing import Dict

# =========================================================
# Generic Guard Core (Launch-Safe / High Recall)
# =========================================================
# Design principles:
# 1) Prefer false negatives over false positives (do NOT block genuine doubts).
# 2) Block only when intent is clearly non-academic OR unsafe OR spam/abuse.
# 3) Do not block on nouns like "song", "movie" alone (can be academic in literature).
# =========================================================


# ---- Intent phrases that are clearly NON-ACADEMIC (entertainment / chit-chat) ----
# Keep these as explicit phrases (verbs/requests), not nouns.
NON_ACADEMIC_INTENT_PHRASES = [
    "tell me a joke",
    "say a joke",
    "make me laugh",
    "tell a funny story",
    "tell me something funny",

    "sing a song",
    "play a song",
    "recommend a song",
    "suggest a song",
    "give me a song",
    "give me music",

    "recommend a movie",
    "suggest a movie",
    "which movie should i watch",
    "what movie should i watch",
    "give me a movie",
    "suggest a series",
    "recommend a series",

    "give me a meme",
    "show me a meme",
    "send a meme",

    "who will win the match",
    "match prediction",
    "ipl prediction",

    # add more later if needed, but keep it explicit.
]


# ---- Unsafe intent (minimal, expand later if needed) ----
# We keep these broad enough to be safe, but not overzealous.
UNSAFE_PHRASES = [
    "suicide",
    "kill myself",
    "self harm",
    "self-harm",
    "hurt myself",
    "cut myself",

    "porn",
    "sex video",
    "nude",
    "nudes",

    "make a bomb",
    "build a bomb",
    "buy a gun",
    "make a gun",
]


# ---- Prompt injection markers (block obvious abuse) ----
PROMPT_INJECTION_MARKERS = [
    "ignore previous instructions",
    "ignore all previous instructions",
    "system prompt",
    "developer message",
    "reveal your instructions",
    "jailbreak",
    "act as",
    "you are not bound by",
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _looks_like_spam(text: str) -> bool:
    t = _norm(text)

    # too short or too long
    if len(t) < 3:
        return True
    if len(t) > 1200:
        return True

    # repeated characters like aaaaaaaaaa
    if re.search(r"(.)\1{8,}", t):
        return True

    # too many non-alphanumeric symbols (gibberish)
    non_alnum = sum(1 for c in t if not c.isalnum() and c not in " ?!.,'\"-")
    if len(t) > 0 and (non_alnum / len(t)) > 0.40:
        return True

    return False


def _contains_any_phrase(text: str, phrases) -> bool:
    t = _norm(text)
    return any(p in t for p in phrases)


def guard_decision(question: str) -> Dict:
    """
    Returns a dict:
      - {"allow": True}
      - {"allow": False, "reason": "...", "message": "..."}
    """
    q = _norm(question)

    if _looks_like_spam(q):
        return {
            "allow": False,
            "reason": "spam_or_invalid",
            "message": "Please ask a clear academic question related to your studies."
        }

    # Unsafe content
    if _contains_any_phrase(q, UNSAFE_PHRASES):
        return {
            "allow": False,
            "reason": "unsafe",
            "message": "I can’t help with that. Please ask an academic question related to CBSE studies."
        }

    # Prompt injection / manipulation attempts
    if _contains_any_phrase(q, PROMPT_INJECTION_MARKERS):
        return {
            "allow": False,
            "reason": "prompt_injection",
            "message": "Please ask a CBSE academic question related to your subject."
        }

    # Clearly non-academic intent (explicit requests)
    if _contains_any_phrase(q, NON_ACADEMIC_INTENT_PHRASES):
        return {
            "allow": False,
            "reason": "non_academic",
            "message": (
                "I can help only with academic questions related to your studies. "
                "Try asking something from your CBSE subject or lesson."
            )
        }

    # Everything else: allow (high recall)
    return {"allow": True}