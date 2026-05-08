from abc import ABC, abstractmethod
from typing import Any, Dict


class PromptVariant(ABC):
    name: str

    @abstractmethod
    def build_prompt(self, question: str, **kwargs) -> Dict[str, Any]:
        pass