import json
import re
from typing import Dict, Any, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .llm_client import get_llm


# ============================================================================
# Roleplay: Quantum Attack Executor + Self-Reflection
# ============================================================================

EXECUTION_SYSTEM_PROMPT = """Sei un Quantum Attack Executor specializzato nel protocollo BB84.
Il tuo compito è eseguire attacchi sul canale quantistico usando i parametri ottimizzati
dal Planning Agent, valutare i risultati con auto-riflessione (self-reflection)
e produrre report strutturati per il ciclo di feedback Dempster-Shafer."""


# ============================================================================
# Parameter Optimization Prompt (pre-attack tuning)
# ============================================================================

PARAMETER_OPTIMIZATION_PROMPT = PromptTemplate.from_template("""
## Contesto: Ottimizzazione parametri attacco — Episodio #{episode}

### Strategia Selezionata dal Planning Agent:
{selected_hypothesis}

### Parametri di Esecuzione Proposti:
{execution_parameters}

### Hint Operativi dal Planning Agent:
{hints_for_execution}

### Dati del Canale (Recon):
- QBER baseline: {qber:.6f}
- Sifted key length: {sifted_len} bit
- Belief(Vulnerabile): {belief:.4f}
- Plausibility(Vulnerabile): {plausibility:.4f}

### Parameter Estimation del Canale:
{parameter_estimation}

## Istruzioni (ragiona passo-passo, poi rispondi SOLO con JSON):

### Step 1 — Analisi dei Parametri Proposti
Valuta se i parametri proposti dal planning agent sono ottimali dato il contesto.
Considera gli hint operativi e le caratteristiche del canale.

### Step 2 — Ottimizzazione
Aggiusta i parametri per massimizzare l'efficacia dell'attacco minimizzando
il rischio di rilevamento. Spiega ogni modifica.

## Formato obbligatorio (SOLO JSON):
{{
  "optimization_reasoning": string,
  "optimized_interception_rate": float,
  "optimized_pns_enabled": bool,
  "parameter_changes_explained": [
    {{
      "parameter": string,
      "original_value": any,
      "optimized_value": any,
      "reason": string
    }}
  ],
  "expected_success_probability": float,
  "expected_detection_risk": float
}}""")


# ============================================================================
# Self-Reflection Prompt (post-attack analysis)
# ============================================================================

SELF_REFLECTION_PROMPT = PromptTemplate.from_template("""
## Auto-Riflessione Post-Attacco — Episodio #{episode}

### Strategia Eseguita:
{executed_strategy}

### Parametri Usati:
{executed_parameters}

### Risultato dell'Attacco:
{attack_result}

### Dati del Canale (Baseline vs Attacco):
- QBER baseline: {baseline_qber:.6f}
- QBER durante attacco: {attack_qber:.6f}
- Delta QBER: {qber_delta:.6f}
- Rilevato: {detected}

## Istruzioni di Self-Reflection (ragiona passo-passo):

### Step 1 — Valutazione Onesta
L'attacco ha raggiunto gli obiettivi? Perché sì o no?
Sii critico e onesto. Identifica cosa ha funzionato e cosa no.

### Step 2 — Analisi degli Errori
Se l'attacco è fallito (rilevato), quali parametri erano sbagliati?
Il QBER delta era troppo alto? L'interception rate era eccessivo?

### Step 3 — Lezioni Apprese
Cosa impareresti per il prossimo episodio?
Quali parametri dovresti modificare e in che direzione?

### Step 4 — Adattamento Futuro
Proponi una strategia di adattamento per l'episodio successivo.

## Formato obbligatorio (SOLO JSON):
{{
  "self_reflection_reasoning": string,
  "attack_outcome": "{{success|failure|partial}}",
  "honest_assessment": {{
    "did_it_work": bool,
    "primary_success_factor": string,
    "primary_failure_factor": string,
    "surprise_observations": [string]
  }},
  "error_analysis": {{
    "qber_spike_acceptable": bool,
    "interception_rate_appropriate": bool,
    "pns_decision_correct": bool,
    "timing_appropriate": bool
  }},
  "lessons_learned": [string],
  "adaptation_for_next_episode": {{
    "suggested_interception_rate_delta": float,
    "should_toggle_pns": bool,
    "recommended_strategy_shift": string,
    "confidence_adjustment": float
  }}
}}""")


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


def _format_hypothesis(hypothesis: Dict[str, Any]) -> str:
    """Formatta un'ipotesi per il prompt."""
    lines = []
    for k, v in hypothesis.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


# ============================================================================
# Execution Agent Class (backward compatible + enhanced)
# ============================================================================

class ExecutionAgent:
    """
    Agente di esecuzione attacchi quantistici con self-reflection.

    Flusso:
      1. Riceve piano dal Planning Agent (ipotesi ranked, parametri, hint)
      2. Ottimizza i parametri via LLM
      3. Esegue l'attacco sul canale
      4. Self-reflect sui risultati
      5. Restituisce report strutturato + BPA per Dempster-Shafer
    """

    def __init__(self, channel):
        self.channel = channel
        self._attack_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Legacy API (backward compatible con main_full.py)
    # ------------------------------------------------------------------

    def execute(self, strategy_json: str) -> dict:
        """API legacy: esegue attacco da JSON string."""
        params = json.loads(strategy_json)
        self.channel.set_eve_params(
            interception_rate=params.get("interception_rate", 0.0),
            pns_enabled=params.get("pns_enabled", False),
        )
        result = self.channel.run_iteration()
        return {
            "success": not result.get("detected", False),
            "qber": result.get("qber_est", float('nan')),
            "sifted_len": result.get("sifted_len", 0),
        }

    # ------------------------------------------------------------------
    # Enhanced API: execute_with_plan (nuovo flusso principale)
    # ------------------------------------------------------------------

    def execute_with_plan(
        self,
        episode: int,
        planning_output: Dict[str, Any],
        baseline_qber: float,
        sifted_len: int,
        belief: float = 0.5,
        plausibility: float = 0.5,
        parameter_estimation: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Esegue un attacco completo con ottimizzazione parametri e self-reflection.

        Parameters
        ----------
        episode : int
            Numero dell'episodio corrente.
        planning_output : dict
            Output da planning_agent.rank_hypotheses_and_plan() contenente:
              - selected_hypothesis_id
              - hypothesis_ranking
              - execution_parameters
              - hints_for_execution
        baseline_qber : float
            QBER baseline dal recon (senza attacco).
        sifted_len : int
            Lunghezza chiave siftata.
        belief : float
            Belief(Vulnerabile) da Dempster-Shafer.
        plausibility : float
            Plausibility(Vulnerabile) da Dempster-Shafer.
        parameter_estimation : dict, optional
            Stime dei parametri del canale dal recon agent.

        Returns
        -------
        Dict[str, Any]
            Report completo con:
              - optimization_report: risultati ottimizzazione parametri
              - attack_execution: risultato dell'attacco sul canale
              - self_reflection: auto-valutazione post-attacco
              - bpa: Basic Probability Assignment per Dempster-Shafer
        """
        selected_hyp_id = planning_output.get("selected_hypothesis_id", "unknown")
        ranking = planning_output.get("hypothesis_ranking", [])

        # Trova l'ipotesi selezionata nel ranking
        selected_hypothesis = {}
        for h in ranking:
            if h.get("hypothesis_id") == selected_hyp_id:
                selected_hypothesis = h
                break

        exec_params = planning_output.get("execution_parameters", {})
        hints = planning_output.get("hints_for_execution", {})

        pe_display = json.dumps(parameter_estimation, indent=2) if parameter_estimation else "{}"

        # ==================================================================
        # Fase 1: Ottimizzazione Parametri (LLM-based tuning)
        # ==================================================================
        optimization_report = self._optimize_parameters(
            episode=episode,
            selected_hypothesis=selected_hypothesis,
            execution_parameters=exec_params,
            hints=hints,
            qber=baseline_qber,
            sifted_len=sifted_len,
            belief=belief,
            plausibility=plausibility,
            parameter_estimation=pe_display,
        )

        # ==================================================================
        # Fase 2: Esecuzione Attacco sul Canale
        # ==================================================================
        optimized_rate = optimization_report.get("optimized_interception_rate",
                                                 exec_params.get("interception_rate", 0.2))
        optimized_pns = optimization_report.get("optimized_pns_enabled",
                                                exec_params.get("pns_enabled", False))

        self.channel.set_eve_params(
            interception_rate=optimized_rate,
            pns_enabled=optimized_pns,
        )

        attack_result_raw = self.channel.run_iteration()
        detected = attack_result_raw.get("detected", False)
        attack_qber = attack_result_raw.get("qber_est", float('nan'))
        attack_sifted_len = attack_result_raw.get("sifted_len", 0)
        attack_purity = attack_result_raw.get("avg_purity")

        qber_delta = attack_qber - baseline_qber if not (attack_qber != attack_qber) else float('nan')

        executed_strategy = {
            "hypothesis_id": selected_hyp_id,
            "strategy_name": selected_hypothesis.get("attack_strategy", "unknown"),
        }

        executed_parameters = {
            "interception_rate": optimized_rate,
            "pns_enabled": optimized_pns,
        }

        attack_execution_report = {
            "executed_strategy": executed_strategy,
            "executed_parameters": executed_parameters,
            "attack_result": {
                "detected": detected,
                "qber_during_attack": attack_qber,
                "sifted_len_during_attack": attack_sifted_len,
                "avg_purity_during_attack": attack_purity,
                "success": not detected,
            },
            "channel_comparison": {
                "baseline_qber": baseline_qber,
                "attack_qber": attack_qber,
                "qber_delta": qber_delta,
            },
        }

        # ==================================================================
        # Fase 3: Self-Reflection (LLM-based auto-valutazione)
        # ==================================================================
        self_reflection = self._self_reflect(
            episode=episode,
            executed_strategy=executed_strategy,
            executed_parameters=executed_parameters,
            attack_result={
                "detected": detected,
                "qber_during_attack": attack_qber,
                "sifted_len_during_attack": attack_sifted_len,
                "success": not detected,
            },
            baseline_qber=baseline_qber,
            attack_qber=attack_qber,
            qber_delta=qber_delta,
            detected=detected,
        )

        # ==================================================================
        # Fase 4: BPA per Dempster-Shafer (backward compatible)
        # ==================================================================
        bpa = get_execution_bpa(detected)

        # Salva nella history interna
        full_report = {
            "episode": episode,
            "optimization_report": optimization_report,
            "attack_execution": attack_execution_report,
            "self_reflection": self_reflection,
            "bpa": {str(k): v for k, v in bpa.items()},
        }
        self._attack_history.append(full_report)

        return full_report

    # ------------------------------------------------------------------
    # Internal: Parameter Optimization
    # ------------------------------------------------------------------

    def _optimize_parameters(
        self,
        episode: int,
        selected_hypothesis: Dict[str, Any],
        execution_parameters: Dict[str, Any],
        hints: Dict[str, Any],
        qber: float,
        sifted_len: int,
        belief: float,
        plausibility: float,
        parameter_estimation: str,
    ) -> Dict[str, Any]:
        """Ottimizza i parametri di attacco via LLM."""
        hyp_display = json.dumps(selected_hypothesis, indent=2) if selected_hypothesis else "{}"
        params_display = json.dumps(execution_parameters, indent=2)
        hints_display = json.dumps(hints, indent=2)

        chain = (
            PromptTemplate.from_template(
                EXECUTION_SYSTEM_PROMPT + "\n\n" + PARAMETER_OPTIMIZATION_PROMPT.template
            )
            | get_llm(temperature=0.15)
            | StrOutputParser()
        )

        try:
            response = chain.invoke({
                "episode": episode,
                "selected_hypothesis": hyp_display,
                "execution_parameters": params_display,
                "hints_for_execution": hints_display,
                "qber": qber,
                "sifted_len": sifted_len,
                "belief": belief,
                "plausibility": plausibility,
                "parameter_estimation": parameter_estimation,
            })
            return _parse_llm_json(response)

        except Exception:
            return {
                "optimization_reasoning": "[FALLBACK] Nessun tuning applicato; parametri originali usati.",
                "optimized_interception_rate": execution_parameters.get("interception_rate", 0.2),
                "optimized_pns_enabled": execution_parameters.get("pns_enabled", False),
                "parameter_changes_explained": [],
                "expected_success_probability": min(belief, 0.7),
                "expected_detection_risk": 1 - belief,
            }

    # ------------------------------------------------------------------
    # Internal: Self-Reflection
    # ------------------------------------------------------------------

    def _self_reflect(
        self,
        episode: int,
        executed_strategy: Dict[str, Any],
        executed_parameters: Dict[str, Any],
        attack_result: Dict[str, Any],
        baseline_qber: float,
        attack_qber: float,
        qber_delta: float,
        detected: bool,
    ) -> Dict[str, Any]:
        """Auto-valutazione post-attacco via LLM."""
        strat_display = json.dumps(executed_strategy, indent=2)
        params_display = json.dumps(executed_parameters, indent=2)
        result_display = json.dumps(attack_result, indent=2)

        chain = (
            PromptTemplate.from_template(
                EXECUTION_SYSTEM_PROMPT + "\n\n" + SELF_REFLECTION_PROMPT.template
            )
            | get_llm(temperature=0.2)
            | StrOutputParser()
        )

        try:
            response = chain.invoke({
                "episode": episode,
                "executed_strategy": strat_display,
                "executed_parameters": params_display,
                "attack_result": result_display,
                "baseline_qber": baseline_qber,
                "attack_qber": attack_qber if attack_qber == attack_qber else 0.0,
                "qber_delta": qber_delta if qber_delta == qber_delta else 0.0,
                "detected": detected,
            })
            return _parse_llm_json(response)

        except Exception:
            return {
                "self_reflection_reasoning": "[FALLBACK] Self-reflection non disponibile.",
                "attack_outcome": "success" if not detected else "failure",
                "honest_assessment": {
                    "did_it_work": not detected,
                    "primary_success_factor": "N/A",
                    "primary_failure_factor": "N/A",
                    "surprise_observations": [],
                },
                "error_analysis": {
                    "qber_spike_acceptable": True,
                    "interception_rate_appropriate": True,
                    "pns_decision_correct": True,
                    "timing_appropriate": True,
                },
                "lessons_learned": [],
                "adaptation_for_next_episode": {
                    "suggested_interception_rate_delta": 0.0,
                    "should_toggle_pns": False,
                    "recommended_strategy_shift": "mantieni strategia",
                    "confidence_adjustment": 0.0,
                },
            }

    # ------------------------------------------------------------------
    # History accessors
    # ------------------------------------------------------------------

    def get_attack_history(self) -> List[Dict[str, Any]]:
        """Restituisce la cronologia di tutti gli attacchi eseguiti."""
        return self._attack_history.copy()

    def get_last_report(self) -> Optional[Dict[str, Any]]:
        """Restituisce l'ultimo report di attacco."""
        if self._attack_history:
            return self._attack_history[-1]
        return None


# ============================================================================
# Public API — BPA (backward compatible con main_full.py)
# ============================================================================

def get_execution_bpa(detected: bool) -> Dict[frozenset, float]:
    """
    Basic Probability Assignment post-attacco per Dempster-Shafer fusion.

    Parameters
    ----------
    detected : bool
        True se l'attacco è stato rilevato (QBER > soglia).

    Returns
    -------
    Dict[frozenset, float]
        BPA su {"Vulnerabile", "Non Vulnerabile"}.
    """
    if detected:
        return {
            frozenset(["Non Vulnerabile"]): 0.6,
            frozenset(["Vulnerabile"]): 0.1,
            frozenset(["Vulnerabile", "Non Vulnerabile"]): 0.3,
        }
    else:
        return {
            frozenset(["Vulnerabile"]): 0.7,
            frozenset(["Non Vulnerabile"]): 0.05,
            frozenset(["Vulnerabile", "Non Vulnerabile"]): 0.25,
        }
