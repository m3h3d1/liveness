"""Final decision from challenge reasons and the passive score."""
PASS_SCORE = 0.7
FAIL_SCORE = 0.3


def decide(reasons, passive):
    reasons = list(reasons)
    if passive is None and "LOW_QUALITY" not in reasons:
        reasons.append("LOW_QUALITY")
    elif passive < FAIL_SCORE:
        reasons.append("PASSIVE_LOW")
    if reasons:
        return "FAIL", reasons
    if passive >= PASS_SCORE:
        return "PASS", []
    return "UNCERTAIN", ["PASSIVE_BORDERLINE"]
