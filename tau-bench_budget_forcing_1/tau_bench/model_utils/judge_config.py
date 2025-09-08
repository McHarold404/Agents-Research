# tau_bench/model_utils/judge_config.py
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class JudgeConfig:
    """Configuration for LLM judge inference"""
    
    # Model and API settings
    base_url: str = "http://localhost:3001/v1"  # User model port
    api_key: str = "EMPTY"
    model: str = "Qwen2.5-72B-Instruct"
    
    # Judge-specific parameters
    temperature: float = 0.1  # Low temperature for consistent judgment
    max_tokens: int = 100     # Shorter response for judgment
    
    # Action selection parameters
    num_candidates: int = 5   # Number of action candidates to generate
    conversation_turns: int = 2  # Number of previous turns to include
    
    # Additional judge parameters
    judge_params: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize judge_params if not provided"""
        if self.judge_params is None:
            self.judge_params = {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
    
    def get_judge_params(self) -> Dict[str, Any]:
        """Get judge parameters for API calls"""
        return self.judge_params.copy()
    
    def update_judge_params(self, **kwargs) -> None:
        """Update judge parameters"""
        self.judge_params.update(kwargs)

# Default judge configuration
DEFAULT_JUDGE_CONFIG = JudgeConfig()

# Alternative configurations for different use cases
FAST_JUDGE_CONFIG = JudgeConfig(
    temperature=0.05,
    max_tokens=50,
    num_candidates=3,
    conversation_turns=1
)

DETAILED_JUDGE_CONFIG = JudgeConfig(
    temperature=0.15,
    max_tokens=200,
    num_candidates=7,
    conversation_turns=3
)