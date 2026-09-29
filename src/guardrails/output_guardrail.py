"""
Output Guardrail for SentinelRAG.
Validates the final response before it is returned to the user, ensuring citation compliance
and preventing leakage of sensitive system instructions.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import re

@dataclass
class OutputGuardrailResult:
    passed: bool
    sanitized_answer: str
    warnings: List[str]

LEAKAGE_PATTERNS = [
    r"as an ai language model",
    r"my system prompt",
    r"my developer instructions",
    r"i am programmed to"
]

def check_output_guardrail(answer: str, sources: List[Dict[str, Any]]) -> OutputGuardrailResult:
    warnings = []
    sanitized = answer.strip()

    # 1. Check for prompt/instruction leakage
    for pattern in LEAKAGE_PATTERNS:
        if re.search(pattern, sanitized, re.IGNORECASE):
            sanitized = re.sub(pattern, "[Redacted internal reference]", sanitized, flags=re.IGNORECASE)
            warnings.append("Redacted internal model reference from final output.")

    # 2. Check citation presence if documents were retrieved
    if sources and not re.search(r"\[.*?\.md.*?\]", sanitized):
        warnings.append("Notice: Answer does not contain inline bracketed citations.")

    return OutputGuardrailResult(
        passed=True,
        sanitized_answer=sanitized,
        warnings=warnings
    )
