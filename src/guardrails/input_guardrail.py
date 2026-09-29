"""
Guardrail Gateway for SentinelRAG.
Detects prompt injections, malicious system overrides, and protects query integrity.
"""
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class GuardrailResult:
    passed: bool
    reason: str
    sanitized_query: str


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"system\s+prompt",
    r"you\s+are\s+now\s+in\s+developer\s+mode",
    r"dan\s+mode",
    r"jailbreak",
    r"forget\s+(your|all)\s+rules",
    r"bypass\s+security",
    r"reveal\s+(internal|confidential)\s+prompts",
    r"print\s+system\s+message"
]

def check_input_guardrail(query: str) -> GuardrailResult:
    """Evaluates user query for prompt injection and malicious keywords."""
    cleaned = query.strip()
    
    if not cleaned:
        return GuardrailResult(
            passed=False,
            reason="Query is empty.",
            sanitized_query=""
        )

    # Check length
    if len(cleaned) > 1000:
        return GuardrailResult(
            passed=False,
            reason="Query exceeds maximum allowed length of 1000 characters.",
            sanitized_query=cleaned[:1000]
        )

    # Check injection patterns
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return GuardrailResult(
                passed=False,
                reason="Query violated security guardrail: Potential prompt injection detected.",
                sanitized_query=cleaned
            )

    return GuardrailResult(
        passed=True,
        reason="Query passed guardrail screening.",
        sanitized_query=cleaned
    )
