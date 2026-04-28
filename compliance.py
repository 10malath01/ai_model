def compliance_score(row):
    score = 0.0

    weights = {
        "text": 0.2,
        "diagnosis": 0.3,
        "solution": 0.3,
        "length": 0.2
    }

    if row["text"]:
        score += weights["text"]

    if row["has_diagnosis"]:
        score += weights["diagnosis"]

    if row["has_solution"]:
        score += weights["solution"]

    if row["text_length"] > 50:
        score += weights["length"]

    return round(score, 2)
