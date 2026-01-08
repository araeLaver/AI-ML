# Prompts Module
from .templates import (
    SYSTEM_PROMPT,
    FRAUD_ANALYSIS_PROMPT,
    FEW_SHOT_EXAMPLES,
    COT_PROMPT,
    get_fraud_analysis_prompt,
)
from .tools import FINANCIAL_TOOLS

__all__ = [
    "SYSTEM_PROMPT",
    "FRAUD_ANALYSIS_PROMPT",
    "FEW_SHOT_EXAMPLES",
    "COT_PROMPT",
    "get_fraud_analysis_prompt",
    "FINANCIAL_TOOLS",
]
