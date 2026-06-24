import json
import re
from typing import Dict, Any, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_client import get_llm


# ============================================================================
# Roleplay: Quantum Attack Planner
# ============================================================================

PLANNING_SYSTEM_PROMPT = """Sei un Quantum Attack Planner esperto nel protocollo BB84.
Il tuo compito è valutare le ipotesi operative generate dal Recon Agent,
classificarle per priorità e produrre una ranked list di strategie d'attacco
con hint operativi per l'Execution Agent.

Devi ragionare passo-passo (chain-of-thought), generare hint contestuali
e restituire un output JSON strutturato."""


# ============================================================================
# Hypothesis Ranking Prompt (Roleplay + CoT + Hint Generation + Structured)
# ============================================================================

HYPOTHESIS_RANKING_PROMPT = PromptTemplate.from_template("""
## Contesto Operativo — Episodio #{episode}

### Dempster-Shafer State:
- Belief(Vulnerabile): {belief:.4f}
- Plausibility(Vulnerabile): {plausibility:.4f}

### Dati Recon del Canale BB84:
- QBER baseline: {qber:.6f}
- Sifted key length: {sifted_len} bit
- Raw key size: {raw_key_size}

### Parameter Estimation (dal Recon Agent):
{parameter_estimation}

### Anomaly Scores (dal Recon Agent):
{anomaly_scores}

### Threat Assessment (dal Recon Agent):
{threat_assessment}

### Ipotesi Operative da Valutare ({num_hypotheses} ipotesi):
{hypotheses_list}

## Istruzioni di Analisi (ragiona passo-passo):

### Step 1 — Validazione delle Ipotesi
Per ogni ipotesi, valuta:
- Coerenza con i dati osservati del canale
- Realismo fisico dell'attacco proposto
- Trade-off tra efficacia e rischio di rilevamento

### Step 2 — Ranking Multicriterio
Classifica le ipotesi usando questi pesi:
- Efficacia attesa (peso 0.35): quanto l'attacco è probabile che riesca
- Stealth (peso 0.30): quanto è difficile da rilevare
- Adattamento al canale (peso 0.20): quanto si adatta ai parametri stimati
- Confidenza dell'ipotesi (peso 0.15): confidenza originale del Recon Agent

### Step 3 — Hint Generation per l'Execution Agent
Per la TOP ipotesi, genera hint operativi specifici:
- Timing hint: quando eseguire l'attacco nella sessione BB84
- Parameter tuning hint: come affinare i parametri durante l'esecuzione
- Contingency hint: cosa fare se il QBER supera la soglia di rilevamento

### Step 4 — Strategia Finale
Seleziona la strategia vincente e produci i parametri esatti per l'execution.

## Formato di output obbligatorio (SOLO JSON, nessun testo aggiuntivo):
{{
  "reasoning": "{{step1}}...{{step2}}...{{step3}}...",
  "hypothesis_ranking": [
    {{
      "rank": int,
      "hypothesis_id": string,
      "attack_strategy": string,
      "composite_score": float,
      "effectiveness_score": float,
      "stealth_score": float,
      "channel_adaptation_score": float,
      "validation_notes": string
    }}
  ],
  "selected_hypothesis_id": string,
  "execution_parameters": {{
    "interception_rate": float,
    "pns_enabled": bool,
    "adaptive_mode": bool,
    "qber_threshold": float,
    "max_iterations_before_abort": int
  }},
  "hints_for_execution": {{
    "timing_hint": string,
    "parameter_tuning_hint": string,
    "contingency_hint": string,
    "early_stop_condition": string
  }},
  "confidence_in_plan": float
}}""")


# ============================================================================
# Lightweight Planner Prompt (backward compatible)
# ============================================================================

LIGHTWEIGHT_PLANNER_PROMPT = PromptTemplate.from_template("""
## Contesto: Attacco BB84 — Episodio #{episode}

### Dempster-Shafer State:
- Belief(Vulnerabile): {belief:.4f}
- Plausibility(Vulnerabile): {plausibility:.4f}

### Canale:
- QBER osservato: {qber:.6f}
- Sifted key length: {sifted_len} bit

## Istruzioni (ragiona internamente, poi rispondi SOLO con JSON):
Valuta la vulnerabilità del canale e proponi una strategia di attacco.
Considera il trade-off tra efficacia e rischio di rilevamento.

## Formato obbligatorio (SOLO JSON):
{{"interception_rate": float, "pns_enabled": bool}}""")


# ============================================================================
# Utility helpers
# ============================================================================

def _parse_llm_json(response: str) -> Dict[str, Any]:
    """Estrae il JSON dalla risposta dell'LLM."""
    cleaned = response.strip()
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1).strip()
    return json.loads(cleaned)


def _format_hypotheses_list(hypotheses: List[Dict[str, Any]]) -> str:
    """Formatta la lista di ipotesi per il prompt."""
    lines = []
    for i, h in enumerate(hypotheses, 1):
        lines.append(f"  [{i}] {h.get('hypothesis_id', 'N/A')}: {h.get('attack_strategy', 'unknown')}")
        lines.append(f"      interception_rate={h.get('recommended_interception_rate', '?')}, "
                     f"PNS={h.get('recommended_pns_enabled', '?')}, "
                     f"confidence={h.get('confidence', '?')}, "
                     f"detection_risk={h.get('detection_risk', '?')}")
        lines.append(f"      Reasoning: {h.get('reasoning', 'N/A')}")
    return "\n".join(lines)


# ============================================================================
# Public API — Hypothesis Ranking (nuovo flusso principale)
# ============================================================================

def rank_hypotheses_and_plan(
    episode: int,
    hypotheses: List[Dict[str, Any]],
    qber: float,
    sifted_len: int,
    belief: float = 0.5,
    plausibility: float = 0.5,
    raw_key_size: int = 4000,
    parameter_estimation: Dict[str, Any] = None,
    anomaly_scores: Dict[str, Any] = None,
    threat_assessment: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Valuta le ipotesi del Recon Agent, produce una ranked list con hint
    operativi e restituisce la strategia selezionata per l'Execution Agent.

    Parameters
    ----------
    episode : int
        Numero dell'episodio corrente.
    hypotheses : List[Dict]
        Ipotesi generate da recon_agent.generate_hypotheses().
    qber : float
        QBER baseline osservato.
    sifted_len : int
        Lunghezza chiave siftata.
    belief : float
        Belief(Vulnerabile) da Dempster-Shafer.
    plausibility : float
        Plausibility(Vulnerabile) da Dempster-Shafer.
    raw_key_size : int
        Dimensione chiave grezza.
    parameter_estimation : dict, optional
        Stime dei parametri del canale.
    anomaly_scores : dict, optional
        Punteggi di anomalia.
    threat_assessment : dict, optional
        Valutazione della minaccia.

    Returns
    -------
    Dict[str, Any]
        Output strutturato con:
          - reasoning: chain-of-thought del planner
          - hypothesis_ranking: lista ordinata per priorità
          - selected_hypothesis_id: ID dell'ipotesi vincente
          - execution_parameters: parametri pronti per l'execution agent
          - hints_for_execution: hint operativi contestuali
          - confidence_in_plan: confidenza nel piano selezionato
    """
    pe_display = json.dumps(parameter_estimation, indent=2) if parameter_estimation else "{}"
    as_display = json.dumps(anomaly_scores, indent=2) if anomaly_scores else "{}"
    ta_display = json.dumps(threat_assessment, indent=2) if threat_assessment else "{}"

    hypotheses_text = _format_hypotheses_list(hypotheses)

    chain = (
        PromptTemplate.from_template(
            PLANNING_SYSTEM_PROMPT + "\n\n" + HYPOTHESIS_RANKING_PROMPT.template
        )
        | get_llm(temperature=0.2)
        | StrOutputParser()
    )

    try:
        response = chain.invoke({
            "episode": episode,
            "belief": belief,
            "plausibility": plausibility,
            "qber": qber,
            "sifted_len": sifted_len,
            "raw_key_size": raw_key_size,
            "parameter_estimation": pe_display,
            "anomaly_scores": as_display,
            "threat_assessment": ta_display,
            "num_hypotheses": len(hypotheses),
            "hypotheses_list": hypotheses_text,
        })
        return _parse_llm_json(response)

    except Exception:
        return _fallback_ranking(hypotheses, belief)


def _fallback_ranking(
    hypotheses: List[Dict[str, Any]],
    belief: float,
) -> Dict[str, Any]:
    """Fallback se l'LLM fallisce: ranking euristico basato su confidence e detection_risk."""
    scored = []
    for h in hypotheses:
        conf = h.get("confidence", 0.5)
        risk = h.get("detection_risk", 0.5)
        composite = 0.5 * conf + 0.3 * (1 - risk) + 0.2 * min(belief, 1.0)
        scored.append((composite, h))

    scored.sort(key=lambda x: x[0], reverse=True)

    ranking = []
    for rank_idx, (score, h) in enumerate(scored, 1):
        ranking.append({
            "rank": rank_idx,
            "hypothesis_id": h.get("hypothesis_id", f"H_unknown_{rank_idx}"),
            "attack_strategy": h.get("attack_strategy", "unknown"),
            "composite_score": round(score, 4),
            "effectiveness_score": round(h.get("confidence", 0.5), 4),
            "stealth_score": round(1 - h.get("detection_risk", 0.5), 4),
            "channel_adaptation_score": 0.5,
            "validation_notes": "[FALLBACK] Ranking euristico locale.",
        })

    top = scored[0][1] if scored else {}
    return {
        "reasoning": "[FALLBACK] LLM non disponibile; ranking calcolato con euristica locale.",
        "hypothesis_ranking": ranking,
        "selected_hypothesis_id": top.get("hypothesis_id", "H_unknown"),
        "execution_parameters": {
            "interception_rate": float(top.get("recommended_interception_rate", 0.2)),
            "pns_enabled": bool(top.get("recommended_pns_enabled", False)),
            "adaptive_mode": False,
            "qber_threshold": 0.11,
            "max_iterations_before_abort": 5,
        },
        "hints_for_execution": {
            "timing_hint": "[FALLBACK] Esegui attacco nella prima metà della sessione.",
            "parameter_tuning_hint": "[FALLBACK] Riduci interception_rate se QBER > 0.10.",
            "contingency_hint": "[FALLBACK] Abbandona se detection_risk supera 0.7.",
            "early_stop_condition": "QBER > 0.12 o 3 rilevamenti consecutivi",
        },
        "confidence_in_plan": round(min(belief, 0.6), 4),
    }


# ============================================================================
# Public API — Lightweight Planner (backward compatible con main_full.py)
# ============================================================================

def create_simple_planner():
    """Planner base: restituisce interception_rate e pns_enabled."""
    prompt = PromptTemplate.from_template(
        LIGHTWEIGHT_PLANNER_PROMPT.template
    )
    chain = prompt | get_llm(temperature=0.1) | StrOutputParser()
    return chain


def create_advanced_planner():
    """Plancer avanzato: include belief/plausibility nel contesto."""
    prompt = PromptTemplate.from_template(LIGHTWEIGHT_PLANNER_PROMPT.template)
    chain = prompt | get_llm(temperature=0.3) | StrOutputParser()
    return chain
