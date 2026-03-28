from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class LLMProvider(ABC):
    """
    Classe abstraite représentant un provider de modèle de langage.

    Tous les providers doivent implémenter au minimum la méthode generate().
    Cela permet au pipeline d'utiliser différents backends sans changer
    sa logique principale.
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Génère une réponse texte à partir d'un prompt utilisateur.
        """
        raise NotImplementedError