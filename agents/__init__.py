"""
Mòdul Agents - Sistema d'Agents per la Hipòtesi de Singularitat Informacional

Aquest mòdul conté els agents que treballen sobre les seqüències Φ generades
pel Nivell 0 per detectar patrons Pₖ i inferir regles emergents ωₖ del
Camp Hologràfic Primordial.

Agents disponibles:
- PatternDetector: Detecta patrons recurrents Pₖ(Φ) (observable + structural v33)
- StructuralPatternDetector: Detecta patrons estructurals (v33 only)
- RuleInferer: Infereix regles emergents ωₖ
- Validator: Valida la consistència i robustesa de les regles
"""

from .pattern_detector import PatternDetector
from .structural_pattern_detector import StructuralPatternDetector
from .rule_inferer import RuleInferer
from .validator import Validator

__all__ = ['PatternDetector', 'StructuralPatternDetector', 'RuleInferer', 'Validator']
