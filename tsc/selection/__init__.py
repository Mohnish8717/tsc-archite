"""
tsc.selection — Attractor-Repeller Persona Selection Engine.

Public API:
    PersonaSelectionEngine  — main orchestrator (engine.py)
    SelectionResult         — typed output (models.py)
"""

from tsc.selection.engine import PersonaSelectionEngine
from tsc.selection.models import SelectionResult, TensionVector, PersonaPole, EpistemicGap

__all__ = [
    "PersonaSelectionEngine",
    "SelectionResult",
    "TensionVector",
    "PersonaPole",
    "EpistemicGap",
]
