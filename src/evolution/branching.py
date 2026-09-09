def reproduction_number(descendant_counts):
    if not descendant_counts:
        return 0.0
    return sum(descendant_counts) / len(descendant_counts)

def regime_from_reproduction_numbers(r_valid, r_novel, r_error):
    if r_valid > 1 and r_novel >= 1 and r_error < 1:
        return "co_development"
    if r_error > 1 and r_valid < 1 and r_novel < 1:
        return "co_extinction_risk"
    if r_error > 1:
        return "co_degradation"
    return "stagnation_or_transition"
