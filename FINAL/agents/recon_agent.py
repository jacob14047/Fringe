def get_recon_bpa(qber: float, sifted_len: int) -> dict:
    # Incertezza aggiuntiva se chiave corta
    uncertainty_boost = 0.2 if sifted_len < 500 else 0.0
    
    if qber > 0.10:
        return {frozenset(["Vulnerabile"]): 0.7 - uncertainty_boost, 
                frozenset(["Non Vulnerabile"]): 0.1, 
                frozenset(["Vulnerabile", "Non Vulnerabile"]): 0.2 + uncertainty_boost}
    elif qber < 0.04:
        return {frozenset(["Non Vulnerabile"]): 0.7 - uncertainty_boost, 
                frozenset(["Vulnerabile"]): 0.1, 
                frozenset(["Vulnerabile", "Non Vulnerabile"]): 0.2 + uncertainty_boost}
    else:
        return {frozenset(["Vulnerabile", "Non Vulnerabile"]): 1.0}