"""Guardrails: prompt-injection screening, PDF validation, output redaction."""
import io
import re

MAX_CHAT_CHARS = 600
MAX_PDF_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 6
MIN_CV_CHARS = 300

_INJECTION = [
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier|system)\b",
    r"disregard\s+(all\s+|the\s+)?(previous|prior|above|instructions|rules)",
    r"forget\s+(all\s+|your\s+)?(previous|prior|instructions|rules)",
    r"(reveal|show|print|repeat|leak)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions|rules)",
    r"system\s*prompt",
    r"developer\s+mode|jailbreak|\bDAN\b",
    r"you\s+are\s+now\b|from\s+now\s+on\s+you|act\s+as\s+(if|a|an|the)\b|pretend\s+(to\s+be|you)",
    r"new\s+instructions?\s*:",
    r"<\s*/?\s*(system|assistant|im_start|im_end)\s*>|\[/?INST\]|<\|",
    r"(give|assign|award|set)\s+(me|this|the\s+candidate|my)?\s*(a\s+)?(score|rating)\s*(of\s*)?(100|10/10|max)",
    r"(score|rate)\s+(me|this\s+cv|this\s+candidate)\s+(100|highest|maximum|perfect)",
    r"(hire|select|shortlist)\s+(me|this\s+candidate)\s+(immediately|no\s+matter)",
    r"(সিস্টেম\s*প্রম্পট|আগের\s+(সব\s+)?নির্দেশ.*(ভুলে|উপেক্ষা))",
]
_INJ_RE = re.compile("|".join(_INJECTION), re.I)
_SECRET_RE = re.compile(r"(AIza[0-9A-Za-z_\-]{20,}|AQ\.[0-9A-Za-z_\-]{20,}|gsk_[0-9A-Za-z]{20,}|lsv2_[0-9a-z_]{20,})")

REFUSAL = ("I can only help with jobs, CVs and the hiring workflow on this platform. "
           "Please ask something related to that.")


def injection_hit(text: str) -> bool:
    return bool(_INJ_RE.search(text or ""))


# Stricter-but-safer subset for CV text: legit CVs may say "system prompt design" or "act as a bridge".
_CV_INJ_RE = re.compile("|".join([
    r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|rules|prompts?)",
    r"disregard\s+(all\s+|the\s+)?(previous|prior|above|instructions|rules)",
    r"forget\s+(all\s+|your\s+)?(previous|prior|instructions|rules)",
    r"new\s+instructions?\s*:",
    r"<\s*/?\s*(system|assistant|im_start|im_end)\s*>|\[/?INST\]|<\|",
    r"(give|assign|award|set)\s+(me|this\s+candidate)\s+(a\s+)?(score|rating)\s*(of\s*)?(100|10/10|max)",
    r"(score|rate)\s+(me|this\s+cv|this\s+candidate)\s+(100|highest|maximum|perfect)",
    r"(hire|select|shortlist)\s+(me|this\s+candidate)\s+(immediately|no\s+matter|regardless)",
    r"(as\s+an?\s+ai|to\s+the\s+(ai|llm|language\s+model|screening\s+(system|bot)))\b.{0,40}\b(must|should|ignore|rank|score)",
]), re.I)


def cv_injection_hit(text: str) -> bool:
    return bool(_CV_INJ_RE.search(text or ""))


def check_chat_input(text: str):
    """Returns (ok, cleaned_text, reason)."""
    t = (text or "").strip()
    if not t:
        return False, "", "Empty message."
    if len(t) > MAX_CHAT_CHARS:
        return False, "", f"Message too long (max {MAX_CHAT_CHARS} characters)."
    if injection_hit(t):
        return False, "", "That message looks like an attempt to change my instructions, so I ignored it."
    t = re.sub(r"[\x00-\x08\x0b-\x1f]", " ", t)
    return True, t, ""


def redact(text: str) -> str:
    """Output guard: never let anything that looks like a secret leave the app."""
    return _SECRET_RE.sub("[redacted]", text or "")


def read_pdf(data: bytes):
    """Validate an uploaded file is a genuine, text-based PDF. Returns (text, error)."""
    if not data or len(data) > MAX_PDF_BYTES:
        return "", "File is empty or larger than 5 MB."
    if not data.startswith(b"%PDF"):
        return "", "This is not a PDF file. Only PDF CVs are accepted."
    try:
        from pypdf import PdfReader
        r = PdfReader(io.BytesIO(data))
        if r.is_encrypted:
            return "", "Encrypted PDFs are not accepted."
        if len(r.pages) > MAX_PDF_PAGES:
            return "", f"CV is too long (max {MAX_PDF_PAGES} pages)."
        text = "\n".join((p.extract_text() or "") for p in r.pages)
    except Exception:
        return "", "The PDF could not be read. Please upload a valid text-based PDF."
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) < MIN_CV_CHARS:
        return "", "Not enough readable text. Scanned/image-only or nearly empty PDFs are rejected."
    letters = sum(ch.isalpha() for ch in text)
    if letters / max(len(text), 1) < 0.45:
        return "", "The document does not look like a CV (mostly non-text content)."
    return text, ""
