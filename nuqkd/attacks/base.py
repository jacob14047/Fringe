"""
nuqkd.attacks.base
===================
Abstract base class for all eavesdropping / attack agents.

Design principles
-----------------
* **Minimal surface**: an attack agent only needs to implement the two hooks
  ``intercept_quantum`` and ``listen_classical``.  All other methods have
  no-op default implementations so that partial attacks (e.g., passive
  eavesdropping only) work out of the box.

* **Non-destructive by default**: the default ``intercept_quantum`` passes
  all photons through unchanged.  Override to modify states or drop photons.

* **Observable**: every agent exposes ``statistics()`` so the simulation
  framework can quantify the attack's information gain, detection probability,
  etc.

* **Composable**: multiple agents can be chained via ``ChainedAttack``.

Hook points
-----------
``intercept_quantum(photons)``
    Called by the ``QuantumChannel`` before physical noise is applied.
    The agent receives the full list of photons in transit and returns a
    (possibly modified / filtered) list.  Return ``None`` in place of a
    photon to block it.

``listen_classical(msg_type, data, sender, recipient)``
    Called by the ``ClassicalChannel`` for every message.  Read-only —
    the channel ignores the return value.

``on_iteration_start(iteration_id, config)``
    Called at the start of each key-distribution round.

``on_iteration_end(iteration_id, result)``
    Called at the end of each key-distribution round.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, List, Optional

import numpy as np

from nuqkd.core.qubit import Photon
from nuqkd.core.classical import ClassicalMsgType


class BaseAttack(ABC):
    """
    Abstract eavesdropping agent.

    All attack implementations (intercept-and-resend, PNS, Trojan-horse, …)
    inherit from this class.  Pentest AI agents should also subclass it.

    Minimal implementation::

        class MyAttack(BaseAttack):
            name = "my_attack"

            def intercept_quantum(self, photons):
                # do something
                return photons

    Attributes
    ----------
    name : str
        Human-readable identifier.
    enabled : bool
        Master on/off switch; the channel checks this before calling hooks.
    """

    name: str    = "base_attack"
    enabled: bool = True

    def __init__(self) -> None:
        self._iteration:    int  = 0
        self._stats:        Dict = {}
        # Bases announced on the classical channel (filled by listen_classical)
        self.alice_bases_announced: Optional[np.ndarray] = None
        self.bob_bases_announced:   Optional[np.ndarray] = None
        self.sifted_indices:        Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Primary hooks (override these)
    # ------------------------------------------------------------------

    def intercept_quantum(self,
                          photons: List[Optional[Photon]]
                          ) -> List[Optional[Photon]]:
        """
        Intercept / modify photons in the quantum channel.

        Parameters
        ----------
        photons : list of Photon | None
            All photons currently in transit.

        Returns
        -------
        list of Photon | None
            The (possibly modified) photon list.  Return ``None`` in a slot
            to block that photon (simulates Eve absorbing it).

        Default: pass through unchanged.
        """
        return photons

    def listen_classical(self,
                         msg_type: ClassicalMsgType,
                         data: Any,
                         sender: str,
                         recipient: str) -> None:
        """
        Eavesdrop on a classical message (read-only).

        Default implementation caches basis announcements so subclasses
        can call ``super().listen_classical(...)`` to benefit from automatic
        basis tracking.
        """
        if msg_type == ClassicalMsgType.ALICE_BASES:
            self.alice_bases_announced = np.asarray(data)
        elif msg_type == ClassicalMsgType.BOB_BASES:
            self.bob_bases_announced = np.asarray(data)
        elif msg_type == ClassicalMsgType.SIFTED_INDICES:
            self.sifted_indices = np.asarray(data)

    # ------------------------------------------------------------------
    # Lifecycle hooks (optional)
    # ------------------------------------------------------------------

    def on_iteration_start(self, iteration_id: int, config: Any) -> None:
        """Called at the beginning of each key-distribution round."""
        self._iteration = iteration_id
        # Reset per-iteration classical state
        self.alice_bases_announced = None
        self.bob_bases_announced   = None
        self.sifted_indices        = None

    def on_iteration_end(self, iteration_id: int, result: Any) -> None:
        """Called at the end of each key-distribution round."""

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def statistics(self) -> Dict[str, Any]:
        """
        Return a dictionary of attack performance metrics.

        Should be overridden to report attack-specific stats such as:
        * Fraction of bits Eve guessed correctly
        * Number of photons intercepted / blocked
        * PNS success rate
        * QBER introduced
        """
        return dict(self._stats)

    def reset(self) -> None:
        """Reset all accumulated state (call between independent simulations)."""
        self._iteration = 0
        self._stats     = {}
        self.alice_bases_announced = None
        self.bob_bases_announced   = None
        self.sifted_indices        = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, enabled={self.enabled})"


# ---------------------------------------------------------------------------
# Utility: chain multiple attacks
# ---------------------------------------------------------------------------

class ChainedAttack(BaseAttack):
    """
    Applies a sequence of attack agents in order.

    The quantum intercept pipeline is applied sequentially — the output of
    agent n is the input to agent n+1.  Classical eavesdropping is broadcast
    to all agents independently.

    Usage::

        chain = ChainedAttack([PNSAttack(config), InterceptResendAttack(rate=0.1)])
    """

    name = "chained"

    def __init__(self, agents: List[BaseAttack]) -> None:
        super().__init__()
        self.agents = agents

    def intercept_quantum(self, photons):
        for agent in self.agents:
            if agent.enabled:
                photons = agent.intercept_quantum(photons)
        return photons

    def listen_classical(self, msg_type, data, sender, recipient):
        super().listen_classical(msg_type, data, sender, recipient)
        for agent in self.agents:
            if agent.enabled:
                agent.listen_classical(msg_type, data, sender, recipient)

    def on_iteration_start(self, iteration_id, config):
        super().on_iteration_start(iteration_id, config)
        for agent in self.agents:
            agent.on_iteration_start(iteration_id, config)

    def on_iteration_end(self, iteration_id, result):
        for agent in self.agents:
            agent.on_iteration_end(iteration_id, result)

    def statistics(self):
        combined = {}
        for agent in self.agents:
            combined[agent.name] = agent.statistics()
        return combined

    def reset(self):
        super().reset()
        for agent in self.agents:
            agent.reset()
