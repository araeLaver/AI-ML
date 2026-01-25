# Training Module

__all__ = []

# LoRA trainer requires trl
try:
    from .train_lora import (
        FinancialLoRATrainer,
        train_financial_model,
        load_training_config,
    )
    __all__.extend([
        "FinancialLoRATrainer",
        "train_financial_model",
        "load_training_config",
    ])
except ImportError:
    pass

# DPO trainer components (some work without full dependencies)
from .dpo_trainer import (
    DPOTrainingConfig,
    PreferenceDataset,
    PreferencePair,
    FinancialPreferenceGenerator,
    create_dpo_dataset_from_instructions,
)
__all__.extend([
    "DPOTrainingConfig",
    "PreferenceDataset",
    "PreferencePair",
    "FinancialPreferenceGenerator",
    "create_dpo_dataset_from_instructions",
])

try:
    from .dpo_trainer import FinancialDPOTrainer
    __all__.append("FinancialDPOTrainer")
except ImportError:
    pass
