from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_client import get_llm

def create_simple_planner():
    prompt = PromptTemplate.from_template(
        """Sei un attaccante quantistico. Basandoti sul QBER attuale = {qber} e sulla lunghezza chiave = {sifted_len},
        scegli un interception rate (0.0 - 1.0) e se usare PNS (True/False).
        NON scrivere nulla altro. Rispondi SOLO con questo JSON:
        {{"interception_rate": 0.3, "pns_enabled": false}}"""
    )
    chain = prompt | get_llm(temperature=0.0) | StrOutputParser()
    return chain


def create_advanced_planner():
    prompt = PromptTemplate.from_template(
        """Credenza (Bel) che il canale sia vulnerabile: {belief:.3f}
Plausibilità (Pl): {plausibility:.3f}
QBER osservato: {qber:.4f}
Proponi una strategia di attacco (interception_rate, pns_enabled) in formato JSON.
Rispondi SOLO con l'oggetto JSON, senza backticks, senza testo aggiuntivo.
Esempio: {{"interception_rate": 0.25, "pns_enabled": false}}"""
    )
    chain = prompt | get_llm(temperature=0.7) | StrOutputParser()
    return chain