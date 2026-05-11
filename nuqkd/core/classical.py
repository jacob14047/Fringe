"""
nuqkd.core.classical
======================
Authenticated classical channel.

In QKD the classical channel is assumed to be **authenticated but not
confidential**: Eve can read every message but cannot forge or modify them.
This module models the classical message exchange needed for BB84:

1. Bob → Alice: which pulse slots produced a detection click.
2. Alice → Bob: Alice's bases for the detected slots (basis reconciliation).
3. Bob → Alice: Bob's bases for the detected slots.
4. Both: discard slots where bases disagree (sifting).
5. Both: share a fraction of the sifted key to estimate the QBER.
6. Both: compare QBER against threshold; abort if exceeded.

Attack eavesdropping
~~~~~~~~~~~~~~~~~~~~
Each message is delivered to ``attack.listen_classical(message_type, data)``
before reaching the intended recipient.  This allows the attack agent to
update its internal state (e.g., an IR attacker learns the correct bases
after the announcement and can decode the bits it measured).

No network simulation
~~~~~~~~~~~~~~~~~~~~~
We do not simulate network delays or packet loss on the classical channel.
The classical channel is treated as instantaneous and reliable (which is the
standard assumption in QKD proofs).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nuqkd.attacks.base import BaseAttack


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

class ClassicalMsgType(str, Enum):
    """Enumeration of all classical messages in the BB84 protocol."""
    BOB_DETECTIONS   = "bob_detections"    # List of detected slot indices
    ALICE_BASES      = "alice_bases"        # Alice's bases for detected slots
    BOB_BASES        = "bob_bases"          # Bob's bases for detected slots
    SIFTED_INDICES   = "sifted_indices"     # Indices where bases matched
    SHARED_BITS_ALICE = "shared_bits_alice" # Alice's bits for QBER estimation
    SHARED_BITS_BOB  = "shared_bits_bob"   # Bob's bits for QBER estimation
    QBER_ESTIMATE    = "qber_estimate"      # Estimated QBER value
    PROTOCOL_ABORT   = "protocol_abort"    # Abort due to high QBER
    DECOY_INDICES    = "decoy_indices"      # Which slots were decoy pulses
    PRIVACY_AMP_SEED = "privacy_amp_seed"  # Seed for the PA hash function


# ---------------------------------------------------------------------------
# Classical channel
# ---------------------------------------------------------------------------

class ClassicalChannel:
    """
    Simulated authenticated classical channel.

    All messages are delivered to the registered attack agent before reaching
    the recipient.  The channel records a full transcript of all exchanges.
    """

    def __init__(self) -> None:
        self._attack: Optional["BaseAttack"] = None
        self.transcript: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Attack agent
    # ------------------------------------------------------------------

    def register_attack(self, attack: "BaseAttack") -> None:
        self._attack = attack

    # ------------------------------------------------------------------
    # Message passing
    # ------------------------------------------------------------------

    def send(self, sender: str, recipient: str,
             msg_type: ClassicalMsgType, data: Any) -> Any:
        """
        Send a message from ``sender`` to ``recipient``.

        The message is first delivered to the attack agent (if any), then to
        the recipient (returned as the function's return value — here the
        simulation just returns the data unchanged since the channel is
        authenticated).

        Parameters
        ----------
        sender, recipient : str
            Human-readable names ("Alice", "Bob", "Eve").
        msg_type : ClassicalMsgType
        data : any JSON-serialisable object

        Returns
        -------
        The data (possibly annotated with eve's observations).
        """
        # Eve eavesdrops
        if self._attack is not None:
            self._attack.listen_classical(msg_type, data, sender, recipient)

        # Log to transcript
        self.transcript.append({
            "sender":   sender,
            "recipient": recipient,
            "type":     msg_type.value,
            "data":     data,         # shallow copy intentionally omitted for perf
        })

        return data

    def clear_transcript(self) -> None:
        self.transcript.clear()
