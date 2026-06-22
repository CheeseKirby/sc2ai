"""Small state tracker for high-level army policy features."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ArmyMemory:
    """Bookkeeping for army-count trends across decision steps."""

    army_count: int = 0
    previous_army_count: int = 0
    attack_army_peak: int = 0
    army_count_delta: int = 0
    army_lost_from_peak: int = 0
    army_lost_from_peak_ratio: float = 0.0
    was_attacking: bool = False

    def update(self, army_count: int, *, is_attacking: bool) -> None:
        """Update memory from the current observed army count."""
        current = max(0, int(army_count))
        self.previous_army_count = self.army_count
        self.army_count = current
        self.army_count_delta = current - self.previous_army_count

        if is_attacking:
            if not self.was_attacking or self.attack_army_peak <= 0:
                self.attack_army_peak = current
            else:
                self.attack_army_peak = max(self.attack_army_peak, current)
        else:
            self.attack_army_peak = 0

        self.was_attacking = bool(is_attacking)
        self._refresh_peak_loss()

    def start_attack(self, army_count: int | None = None) -> None:
        """Initialize attack-phase peak when a policy starts an attack."""
        current = self.army_count if army_count is None else max(0, int(army_count))
        self.attack_army_peak = max(self.attack_army_peak, current)
        self.was_attacking = True
        self._refresh_peak_loss()

    def reset_attack_peak(self) -> None:
        """Clear attack-phase loss state after the retreat row has been recorded."""
        self.attack_army_peak = 0
        self.army_lost_from_peak = 0
        self.army_lost_from_peak_ratio = 0.0
        self.was_attacking = False

    def _refresh_peak_loss(self) -> None:
        peak = max(0, int(self.attack_army_peak))
        lost = max(peak - int(self.army_count), 0)
        self.army_lost_from_peak = lost
        self.army_lost_from_peak_ratio = (lost / peak) if peak else 0.0
