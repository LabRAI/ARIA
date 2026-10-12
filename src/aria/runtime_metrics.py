"""Interpret persisted runtime inspection failures for paired evaluation."""

MALFORMED_ERRORS = {
    "ValueError", "TypeError", "JSONDecodeError", "KeyError", "StructuredGenerationError",
}


def inspection_failure(audit):
    """Distinguish malformed generated predictions from requests that failed."""
    decision = audit.get("intervention") or {}
    error = audit.get("error") or decision.get("error")
    if not error:
        return None
    calls = audit.get("model_calls", [])
    if not calls or not any(call.get("raw_response") for call in calls):
        return "infrastructure"
    error_type = str(decision.get("reason") or error).split(":")[0]
    if error_type in {"gate_error", "repair_error"}:
        error_type = str(decision["reason"]).split(":")[-1]
    return "malformed" if error_type in MALFORMED_ERRORS else "infrastructure"
