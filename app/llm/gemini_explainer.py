def generate_explanation(driver_json):
    """
    Placeholder for Gemini.
    Converts driver JSON to short explanation.
    Must NOT add new facts.
    """

    if driver_json["type"] == "baseline":
        return "Insufficient historical stress signals were observed, so risk is assessed as stable."

    drivers = driver_json["drivers"]
    parts = [f"{d['feature']} contributed to risk" for d in drivers[:2]]

    return "Risk is driven primarily by " + " and ".join(parts) + "."
