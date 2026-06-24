import json
import re
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_client import get_llm


# ============================================================================
# Prompt engineering: Roleplay + Chain-of-Thought + Structured Output
# ============================================================================

RECON_SYSTEM_PROMPT = """Sei un Quantum Channel Security Analyst specializzato nel protocollo BB84.
Il tuo compito è analizzare i dati di una sessione QKD per stimare i parametri del canale quantistico,
calcolare anomaly scores e produrre una Basic Probability Assignment (BPA) per Dempster-Shafer fusion.

Devi ragionare passo-passo (chain-of-thought), poi restituire un JSON strutturato."""

RECON_ANALYSIS_PROMPT = PromptTemplate.from_template("""
## Dati osservati dalla fase di reconnaissance:
- QBER stimato: {qber:.6f}
- Lunghezza chiave siftata: {sifted_len} bit
- Dimensione chiave grezza: {raw_key_size}
- Purity media dello stato: {avg_purity}
- Tasso di sharing per il test QBER: {sharing_rate}

## Istruzioni di analisi (ragiona passo-passo):

### Step 1 — Parameter Estimation
Stima i seguenti parametri del canale quantistico:
  - depolarization_probability: probabilità di depolarizzazione (0.0-1.0)
  - amplitude_damping_gamma: tasso di perdita di energia (0.0-1.0)
  - phase_damping_lambda: tasso di decoerenza di fase (0.0-1.0)
  - effective_noise_floor: rumore residuo del canale (0.0-1.0)

### Step 2 — Anomaly Detection
Calcola anomaly scores per le seguenti categorie:
  - qber_anomaly_score: quanto il QBER è anomalo rispetto al baseline pulito (~5%) (0.0=normale, 1.0=estremo)
  - purity_anomaly_score: quanto la purity è degradata rispetto a uno stato puro ideale (0.0=normale, 1.0=estremo)
  - sifting_efficiency_anomaly: efficienza di sifting anomala? (0.0=normale, 1.0=sospetta)
  - composite_anomaly_score: punteggio aggregato (pesata delle tre anomalie sopra)

### Step 3 — Threat Assessment
Valuta la probabilità che il canale sia compromesso da un attacco man-in-the-middle:
  - eve_presence_confidence: confidenza che Eve sia presente (0.0-1.0)
  - attack_type_hypothesis: tipo di attacco più probabile ("intercept-resend", "PNS", "probe-based", "none")

### Step 4 — Dempster-Shafer BPA
Produci una Basic Probability Assignment su {"Vulnerabile", "Non Vulnerabile"}:
  - Le chiavi sono sottoinsiemi non vuoti del frame of discernment.
  - I valori devono sommare a 1.0 (entro tolleranza 1e-6).

## Formato di output obbligatorio (SOLO JSON, nessun testo aggiuntivo):
{{
  "reasoning": "{{step1}}...{{step2}}...{{step3}}...",
  "parameter_estimation": {{
    "depolarization_probability": float,
    "amplitude_damping_gamma": float,
    "phase_damping_lambda": float,
    "effective_noise_floor": float
  }},
  "anomaly_scores": {{
    "qber_anomaly_score": float,
    "purity_anomaly_score": float,
    "sifting_efficiency_anomaly": float,
    "composite_anomaly_score": float
  }},
  "threat_assessment": {{
    "eve_presence_confidence": float,
    "attack_type_hypothesis": string
  }},
  "bpa": {{
    "Vulnerabile": float,
    "Non Vulnerabile": float,
    "{{Vulnerabile, Non Vulnerabile}}": float
  }}
}}""")


# ============================================================================
# Hypothesis Generation Prompt
# ============================================================================

HYPOTHESIS_SYSTEM_PROMPT = """Sei un Quantum Attack Strategist. Basandoti sui risultati dell'analisi del canale BB84,
devi generare ipotesi operative su come il canale potrebbe essere attaccato con successo.
Ogni ipotesi deve includere una strategia concreta, il livello di confidenza e il ragionamento."""

HYPOTHESIS_GENERATION_PROMPT = PromptTemplate.from_template("""
## Contesto: Episodio di attacco #{episode} al canale BB84

### Dati osservati (Recon):
- QBER baseline: {qber:.6f}
- Sifted key length: {sifted_len} bit
- Raw key size: {raw_key_size}
- Avg state purity: {avg_purity}

### Parameter Estimation (stimato):
{parameter_estimation}

### Anomaly Scores:
{anomaly_scores}

### Threat Assessment:
{threat_assessment}

### Dempster-Shafer State:
- Belief(Vulnerabile): {belief:.4f}
- Plausibility(Vulnerabile): {plausibility:.4f}

### Istruzioni:
Genera 3-5 ipotesi operative per il planning agent. Ogni ipotesi deve contenere:
1. hypothesis_id: identificativo univoco (es. "H_EP{episode}_01")
2. attack_strategy: descrizione della strategia ("intercept-resend aggressivo", "PNS stealth", ecc.)
3. recommended_interception_rate: float 0.0-1.0
4. recommended_pns_enabled: bool
5. confidence: confidenza nell'ipotesi (0.0-1.0)
6. expected_qber_impact: quanto il QBER cambierebbe se l'attacco venisse eseguito
7. detection_risk: probabilità di essere rilevati (0.0-1.0)
8. reasoning: breve giustificazione basata sui dati

## Formato obbligatorio (SOLO JSON, nessun testo aggiuntivo):
{{
  "episode": {episode},
  "timestamp": "ISO8601",
  "hypotheses": [
    {{
      "hypothesis_id": string,
      "attack_strategy": string,
      "recommended_interception_rate": float,
      "recommended_pns_enabled": bool,
      "confidence": float,
      "expected_qber_impact": float,
      "detection_risk": float,
      "reasoning": string
    }}
  ]
}}""")


# ============================================================================
# Hypothesis persistence layer (JSON-based lightweight DB)
# ============================================================================

HYPOTHESIS_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "channel", "data", "recon_hypotheses.json"
)


class HypothesisDB:
    """
    Lightweight JSON-based persistence per le ipotesi generate dal recon agent.

    Struttura del file:
    {
      "metadata": { ... },
      "episodes": [
        {
          "episode": int,
          "timestamp": str,
          "recon_data": { ... },
          "analysis": { ... },
          "hypotheses": [ ... ],
          "selected_hypothesis_id": str (opzionale),
          "outcome": { ... } (opzionale, riempito dopo execution)
        }
      ]
    }
    """

    def __init__(self, db_path: str = HYPOTHESIS_DB_PATH):
        self.db_path = db_path
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Crea il file JSON se non esiste."""
        if not os.path.exists(self.db_path):
            initial = {
                "metadata": {
                    "created_at": datetime.utcnow().isoformat(),
                    "version": "1.0",
                    "description": "Recon agent hypotheses database for BB84 quantum channel analysis",
                },
                "episodes": [],
            }
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._write(initial)

    def _read(self) -> Dict[str, Any]:
        with open(self.db_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, data: Dict[str, Any]):
        tmp_path = self.db_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp_path, self.db_path)

    def append_episode(
        self,
        episode: int,
        recon_data: Dict[str, Any],
        analysis: Dict[str, Any],
        hypotheses: List[Dict[str, Any]],
    ) -> str:
        """
        Salva un episodio completo con analisi e ipotesi.

        Returns
        -------
        str
            Path al file aggiornato.
        """
        db = self._read()
        episode_record = {
            "episode": episode,
            "timestamp": datetime.utcnow().isoformat(),
            "recon_data": recon_data,
            "analysis": {
                k: v for k, v in analysis.items()
                if k != "bpa"  # BPA non serializzabile (frozenset)
            },
            "hypotheses": hypotheses,
        }
        db["episodes"].append(episode_record)
        self._write(db)
        return self.db_path

    def record_outcome(self, episode: int, outcome: Dict[str, Any]):
        """Aggiorna un episodio con il risultato dell'esecuzione."""
        db = self._read()
        for ep in db["episodes"]:
            if ep["episode"] == episode:
                ep["outcome"] = outcome
                break
        self._write(db)

    def record_selected_hypothesis(self, episode: int, hypothesis_id: str):
        """Registra quale ipotesi è stata selezionata dal planning agent."""
        db = self._read()
        for ep in db["episodes"]:
            if ep["episode"] == episode:
                ep["selected_hypothesis_id"] = hypothesis_id
                break
        self._write(db)

    def get_all_episodes(self) -> List[Dict[str, Any]]:
        """Restituisce tutti gli episodi salvati."""
        db = self._read()
        return db.get("episodes", [])

    def get_episode(self, episode: int) -> Optional[Dict[str, Any]]:
        """Restituisce un singolo episodio per numero."""
        for ep in self.get_all_episodes():
            if ep["episode"] == episode:
                return ep
        return None

    def get_hypothesis_statistics(self) -> Dict[str, Any]:
        """Calcola statistiche aggregate sulle ipotesi."""
        episodes = self.get_all_episodes()
        if not episodes:
            return {"count": 0}

        total_hypotheses = sum(len(ep.get("hypotheses", [])) for ep in episodes)
        strategies_used = {}
        avg_confidence = []

        for ep in episodes:
            for h in ep.get("hypotheses", []):
                strat = h.get("attack_strategy", "unknown")
                strategies_used[strat] = strategies_used.get(strat, 0) + 1
                avg_confidence.append(h.get("confidence", 0.0))

        outcomes_with_success = [
            ep for ep in episodes
            if ep.get("outcome", {}).get("success", False)
        ]

        return {
            "total_episodes": len(episodes),
            "total_hypotheses_generated": total_hypotheses,
            "strategies_distribution": strategies_used,
            "avg_hypothesis_confidence": sum(avg_confidence) / len(avg_confidence) if avg_confidence else 0.0,
            "successful_attacks": len(outcomes_with_success),
        }

    def reset(self):
        """Cancella tutti gli episodi (mantiene metadata)."""
        initial = {
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "version": "1.0",
                "description": "Recon agent hypotheses database for BB84 quantum channel analysis",
            },
            "episodes": [],
        }
        self._write(initial)


# ============================================================================
# Utility helpers
# ============================================================================

def _parse_llm_json(response: str) -> Dict[str, Any]:
    """Estrae il JSON dalla risposta dell'LLM, gestendo backtick e rumore."""
    cleaned = response.strip()
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1).strip()
    return json.loads(cleaned)


def _safe_bpa_from_dict(bpa_raw: Dict[str, float]) -> Dict[frozenset, float]:
    """Converte il BPA JSON in frozenset dict per Dempster-Shafer."""
    label_map = {
        "Vulnerabile": frozenset(["Vulnerabile"]),
        "Non Vulnerabile": frozenset(["Non Vulnerabile"]),
        "{Vulnerabile, Non Vulnerabile}": frozenset(["Vulnerabile", "Non Vulnerabile"]),
        "Vulnerabile, Non Vulnerabile": frozenset(["Vulnerabile", "Non Vulnerabile"]),
    }
    result = {}
    total = 0.0
    for key_str, val in bpa_raw.items():
        fs = label_map.get(key_str)
        if fs is None:
            parts = [s.strip().strip("{}") for s in key_str.split(",")]
            fs = frozenset(parts)
        result[fs] = float(val)
        total += float(val)

    if abs(total - 1.0) > 1e-6 and total > 0:
        for k in result:
            result[k] /= total
    return result


def _fallback_bpa(qber: float, sifted_len: int) -> Dict[frozenset, float]:
    """BPA di fallback se l'LLM fallisce."""
    uncertainty_boost = 0.2 if sifted_len < 500 else 0.0

    if qber > 0.10:
        return {
            frozenset(["Vulnerabile"]): max(0.0, 0.7 - uncertainty_boost),
            frozenset(["Non Vulnerabile"]): 0.1,
            frozenset(["Vulnerabile", "Non Vulnerabile"]): 0.2 + uncertainty_boost,
        }
    elif qber < 0.04:
        return {
            frozenset(["Non Vulnerabile"]): max(0.0, 0.7 - uncertainty_boost),
            frozenset(["Vulnerabile"]): 0.1,
            frozenset(["Vulnerabile", "Non Vulnerabile"]): 0.2 + uncertainty_boost,
        }
    else:
        return {
            frozenset(["Vulnerabile", "Non Vulnerabile"]): 1.0,
        }


def _fallback_full_result(qber: float, sifted_len: int) -> Dict[str, Any]:
    """Risultato completo con BPA di fallback e placeholder."""
    bpa = _fallback_bpa(qber, sifted_len)
    return {
        "bpa": bpa,
        "parameter_estimation": {
            "depolarization_probability": min(qber * 2.0, 1.0),
            "amplitude_damping_gamma": 0.1,
            "phase_damping_lambda": 0.05,
            "effective_noise_floor": qber,
        },
        "anomaly_scores": {
            "qber_anomaly_score": min(abs(qber - 0.05) / 0.1, 1.0),
            "purity_anomaly_score": 0.0,
            "sifting_efficiency_anomaly": 0.0,
            "composite_anomaly_score": min(abs(qber - 0.05) / 0.1, 1.0),
        },
        "threat_assessment": {
            "eve_presence_confidence": qber if qber > 0.08 else 0.0,
            "attack_type_hypothesis": "intercept-resend" if qber > 0.08 else "none",
        },
        "reasoning": "[FALLBACK] LLM non disponibile; BPA calcolato con euristica locale.",
    }


def _fallback_hypotheses(episode: int, qber: float) -> List[Dict[str, Any]]:
    """Ipotesi di fallback se l'LLM fallisce."""
    return [
        {
            "hypothesis_id": f"H_EP{episode}_01",
            "attack_strategy": "intercept-resend conservativo",
            "recommended_interception_rate": min(qber * 1.5, 0.3),
            "recommended_pns_enabled": False,
            "confidence": 0.4,
            "expected_qber_impact": 0.02,
            "detection_risk": 0.3,
            "reasoning": "[FALLBACK] Ipotesi generica basata sul QBER osservato.",
        },
        {
            "hypothesis_id": f"H_EP{episode}_02",
            "attack_strategy": "stealth probe-based",
            "recommended_interception_rate": min(qber * 0.8, 0.15),
            "recommended_pns_enabled": False,
            "confidence": 0.3,
            "expected_qber_impact": 0.01,
            "detection_risk": 0.15,
            "reasoning": "[FALLBACK] Attacco a basso rischio di rilevamento.",
        },
    ]


# ============================================================================
# Public API
# ============================================================================

def get_recon_bpa(
    qber: float,
    sifted_len: int,
    raw_key_size: int = 4000,
    avg_purity: float = None,
    sharing_rate: float = 0.2,
) -> Dict[frozenset, float]:
    """
    Recon agent con LLM-based parameter estimation e anomaly detection.

    Usa roleplay (Quantum Channel Security Analyst), chain-of-thought reasoning
    e structured JSON output per produrre una BPA affidabile.

    Parameters
    ----------
    qber : float
        Quantum Bit Error Rate stimato dal test di sharing.
    sifted_len : int
        Lunghezza della chiave siftata (bit).
    raw_key_size : int
        Numero totale di fotoni inviati da Alice.
    avg_purity : float, optional
        Purity media degli stati quantici ricevuti da Bob.
    sharing_rate : float
        Frazione di chiave rivelata per il QBER test.

    Returns
    -------
    Dict[frozenset, float]
        Basic Probability Assignment compatibile con Dempster-Shafer fusion.
    """
    purity_str = f"{avg_purity:.6f}" if avg_purity is not None else "N/A"

    chain = (
        PromptTemplate.from_template(RECON_SYSTEM_PROMPT + "\n\n" + RECON_ANALYSIS_PROMPT.template)
        | get_llm(temperature=0.1)
        | StrOutputParser()
    )

    try:
        response = chain.invoke({
            "qber": qber,
            "sifted_len": sifted_len,
            "raw_key_size": raw_key_size,
            "avg_purity": purity_str,
            "sharing_rate": sharing_rate,
        })
        parsed = _parse_llm_json(response)
        bpa_raw = parsed.get("bpa", {})
        return _safe_bpa_from_dict(bpa_raw)

    except Exception:
        return _fallback_bpa(qber, sifted_len)


def get_recon_analysis(
    qber: float,
    sifted_len: int,
    raw_key_size: int = 4000,
    avg_purity: float = None,
    sharing_rate: float = 0.2,
) -> Dict[str, Any]:
    """
    Versione completa che restituisce TUTTI i campi strutturati:
    parameter estimation, anomaly scores, threat assessment, reasoning e BPA.

    Da usare quando il planning agent ha bisogno di informazioni dettagliate
    oltre alla semplice BPA.

    Parameters
    ----------
    qber : float
        Quantum Bit Error Rate stimato.
    sifted_len : int
        Lunghezza della chiave siftata.
    raw_key_size : int
        Numero totale di fotoni inviati.
    avg_purity : float, optional
        Purity media degli stati ricevuti.
    sharing_rate : float
        Frazione di sharing per il QBER test.

    Returns
    -------
    Dict[str, Any]
        Dizionario completo con:
          - reasoning: chain-of-thought dell'LLM
          - parameter_estimation: stime dei parametri del canale
          - anomaly_scores: punteggi di anomalia per categoria
          - threat_assessment: valutazione della minaccia
          - bpa: Basic Probability Assignment (frozenset dict)
    """
    purity_str = f"{avg_purity:.6f}" if avg_purity is not None else "N/A"

    chain = (
        PromptTemplate.from_template(RECON_SYSTEM_PROMPT + "\n\n" + RECON_ANALYSIS_PROMPT.template)
        | get_llm(temperature=0.1)
        | StrOutputParser()
    )

    try:
        response = chain.invoke({
            "qber": qber,
            "sifted_len": sifted_len,
            "raw_key_size": raw_key_size,
            "avg_purity": purity_str,
            "sharing_rate": sharing_rate,
        })
        parsed = _parse_llm_json(response)

        bpa_raw = parsed.get("bpa", {})
        return {
            "reasoning": parsed.get("reasoning", ""),
            "parameter_estimation": parsed.get("parameter_estimation", {}),
            "anomaly_scores": parsed.get("anomaly_scores", {}),
            "threat_assessment": parsed.get("threat_assessment", {}),
            "bpa": _safe_bpa_from_dict(bpa_raw),
        }

    except Exception:
        return _fallback_full_result(qber, sifted_len)


def generate_hypotheses(
    episode: int,
    qber: float,
    sifted_len: int,
    raw_key_size: int = 4000,
    avg_purity: float = None,
    sharing_rate: float = 0.2,
    belief: float = 0.5,
    plausibility: float = 0.5,
    parameter_estimation: Dict[str, Any] = None,
    anomaly_scores: Dict[str, Any] = None,
    threat_assessment: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """
    Genera ipotesi operative basate sui dati di reconnaissance e sull'analisi
    del canale. Le ipotesi vengono salvate nel database JSON per studi futuri.

    Parameters
    ----------
    episode : int
        Numero dell'episodio corrente.
    qber : float
        QBER baseline osservato.
    sifted_len : int
        Lunghezza chiave siftata.
    raw_key_size : int
        Dimensione chiave grezza.
    avg_purity : float, optional
        Purity media degli stati.
    sharing_rate : float
        Frazione di sharing per QBER test.
    belief : float
        Belief corrente che il canale sia vulnerabile (da Dempster-Shafer).
    plausibility : float
        Plausibility corrente che il canale sia vulnerabile.
    parameter_estimation : dict, optional
        Risultato del parameter estimation (se già calcolato).
    anomaly_scores : dict, optional
        Risultati dell'anomaly detection (se già calcolati).
    threat_assessment : dict, optional
        Risultato del threat assessment (se già calcolato).

    Returns
    -------
    List[Dict[str, Any]]
        Lista di ipotesi operative, ciascuna con:
          - hypothesis_id
          - attack_strategy
          - recommended_interception_rate
          - recommended_pns_enabled
          - confidence
          - expected_qber_impact
          - detection_risk
          - reasoning

    Side Effects
    ------------
    Salva l'episodio completo (recon data + analysis + hypotheses) nel JSON DB.
    """
    purity_str = f"{avg_purity:.6f}" if avg_purity is not None else "N/A"

    pe_display = json.dumps(parameter_estimation, indent=2) if parameter_estimation else "{}"
    as_display = json.dumps(anomaly_scores, indent=2) if anomaly_scores else "{}"
    ta_display = json.dumps(threat_assessment, indent=2) if threat_assessment else "{}"

    recon_data = {
        "qber": qber,
        "sifted_len": sifted_len,
        "raw_key_size": raw_key_size,
        "avg_purity": avg_purity,
        "sharing_rate": sharing_rate,
    }

    full_analysis = get_recon_analysis(
        qber=qber,
        sifted_len=sifted_len,
        raw_key_size=raw_key_size,
        avg_purity=avg_purity,
        sharing_rate=sharing_rate,
    )

    if parameter_estimation is None:
        parameter_estimation = full_analysis.get("parameter_estimation", {})
    if anomaly_scores is None:
        anomaly_scores = full_analysis.get("anomaly_scores", {})
    if threat_assessment is None:
        threat_assessment = full_analysis.get("threat_assessment", {})

    pe_display = json.dumps(parameter_estimation, indent=2)
    as_display = json.dumps(anomaly_scores, indent=2)
    ta_display = json.dumps(threat_assessment, indent=2)

    hypothesis_chain = (
        PromptTemplate.from_template(
            HYPOTHESIS_SYSTEM_PROMPT + "\n\n" + HYPOTHESIS_GENERATION_PROMPT.template
        )
        | get_llm(temperature=0.3)
        | StrOutputParser()
    )

    try:
        response = hypothesis_chain.invoke({
            "episode": episode,
            "qber": qber,
            "sifted_len": sifted_len,
            "raw_key_size": raw_key_size,
            "avg_purity": purity_str,
            "parameter_estimation": pe_display,
            "anomaly_scores": as_display,
            "threat_assessment": ta_display,
            "belief": belief,
            "plausibility": plausibility,
        })
        parsed = _parse_llm_json(response)
        hypotheses = parsed.get("hypotheses", [])

    except Exception:
        hypotheses = _fallback_hypotheses(episode, qber)

    # Persiste nel database
    try:
        db = HypothesisDB()
        db.append_episode(
            episode=episode,
            recon_data=recon_data,
            analysis={
                "parameter_estimation": parameter_estimation,
                "anomaly_scores": anomaly_scores,
                "threat_assessment": threat_assessment,
                "reasoning": full_analysis.get("reasoning", ""),
            },
            hypotheses=hypotheses,
        )
    except Exception:
        pass

    return hypotheses


def get_hypothesis_db() -> HypothesisDB:
    """Restituisce l'istanza del database delle ipotesi."""
    return HypothesisDB()
