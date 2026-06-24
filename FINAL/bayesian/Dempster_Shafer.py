
class DempsterShafer:
    """
    Implementazione della Teoria di Dempster-Shafer per la fusione di evidenze.
    Gestisce il combinazione di mass function da diverse fonti (es. Recon e Planning).
    """

    def __init__(self, frame_of_discernment):
        """
        Inizializza il modulo con il frame of discernment (e.g., ["Vulnerabile", "Non Vulnerabile"]).
        Lo stato iniziale è di completa ignoranza.
        """
        self.frame = frame_of_discernment  # e.g., ["V", "NV"]
        # Insieme dei sottoinsiemi del frame (power set), mappato a stringhe per semplicità.
        self.power_set = self._generate_power_set()
        # Massa di credenza iniziale: tutta la massa è assegnata all'ignoranza (il frame completo).
        self.mass = {frozenset(subset): 0.0 for subset in self.power_set}
        self.mass[frozenset(self.frame)] = 1.0
        self._normalize_mass()

    def _generate_power_set(self):
        """Genera il power set (tutti i sottoinsiemi) del frame."""
        power_set = []
        n = len(self.frame)
        for i in range(1 << n):
            subset = [self.frame[j] for j in range(n) if (i & (1 << j))]
            power_set.append(subset)
        return power_set

    def _normalize_mass(self):
        """Normalizza la mass function per assicurarsi che la somma sia 1."""
        total = sum(self.mass.values())
        if total > 0:
            for k in self.mass:
                self.mass[k] /= total

    def assign_mass(self, bpa, source_name="Source"):
        """
        Assegna una nuova BPA da una fonte (e.g., Recon o Planning).
        La BPA deve essere un dizionario con chiavi come tuple/frozenset e valori float.
        Esempio valido: {frozenset(["V"]): 0.6, frozenset(["NV"]): 0.1, frozenset(["V","NV"]): 0.3}
        """
        if sum(bpa.values()) > 1:
            raise ValueError("La somma delle Basic Probability Assignments non può superare 1.")
        self.mass = self._combine(self.mass, bpa)
        self._normalize_mass()

    def _combine(self, m1, m2):
        """
        Applica la Regola di Combinazione di Dempster per fondere due mass function.
        Questa è l'implementazione della formula matematica vista sopra.
        """
        combined = {frozenset(): 0.0}
        for A in m1:
            for B in m2:
                intersection = A.intersection(B)
                if not intersection:
                    # Insieme vuoto: contribuisce al fattore di conflitto
                    continue
                # Calcola la massa per l'intersezione
                combined[intersection] = combined.get(intersection, 0.0) + m1[A] * m2[B]

        # Calcola e applica il fattore di normalizzazione (1 - conflitto)
        conflict = sum(m1[A] * m2[B] for A in m1 for B in m2 if not A.intersection(B))
        if conflict == 1.0:
            raise ValueError("Conflitto totale tra le evidenze. Impossibile combinarle.")
        for k in combined:
            combined[k] /= (1 - conflict)

        # Rimuovi l'eventuale chiave per l'insieme vuoto
        combined.pop(frozenset(), None)
        return combined

    def get_belief(self, hypothesis):
        """
        Calcola la funzione di credenza `Bel` per una data ipotesi (o insieme di ipotesi).
        `Bel` rappresenta la fiducia totale che la verità risieda nell'ipotesi.
        """
        hyp_set = frozenset(hypothesis)
        belief = 0.0
        for A in self.mass:
            if A.issubset(hyp_set):
                belief += self.mass[A]
        return belief

    def get_plausibility(self, hypothesis):
        """
        Calcola la funzione di plausibilità `Pl` per una data ipotesi.
        `Pl` rappresenta il grado con cui l'evidenza non smentisce l'ipotesi.
        E' l'opposto della credenza nella negazione dell'ipotesi.
        """
        hyp_set = frozenset(hypothesis)
        plausibility = 0.0
        for A in self.mass:
            if A.intersection(hyp_set):
                plausibility += self.mass[A]
        return plausibility

    def get_current_state(self):
        """Restituisce lo stato corrente delle credenze per ogni ipotesi."""
        state = {}
        for subset in self.power_set:
            if subset:
                state[str(subset)] = self.mass.get(frozenset(subset), 0.0)
        return state