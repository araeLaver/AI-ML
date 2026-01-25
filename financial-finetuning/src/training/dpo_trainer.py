# -*- coding: utf-8 -*-
"""
Direct Preference Optimization (DPO) Trainer for Financial LLMs

This module provides DPO training capabilities for fine-tuning LLMs
to align with human preferences in financial domain tasks.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import DPOTrainer as TRLDPOTrainer, DPOConfig
    DPO_AVAILABLE = True
except ImportError:
    DPO_AVAILABLE = False

try:
    from bitsandbytes import BitsAndBytesConfig
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


@dataclass
class PreferencePair:
    """A single preference pair for DPO training.

    Attributes:
        prompt: The input prompt
        chosen: The preferred (better) response
        rejected: The rejected (worse) response
        category: Optional category label
    """
    prompt: str
    chosen: str
    rejected: str
    category: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "category": self.category,
            "metadata": self.metadata,
        }


@dataclass
class DPOTrainingConfig:
    """Configuration for DPO training.

    Attributes:
        model_name: Base model name or path
        output_dir: Directory for checkpoints and outputs
        beta: DPO beta parameter (temperature)
        learning_rate: Learning rate
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        max_length: Maximum sequence length
        max_prompt_length: Maximum prompt length
    """
    model_name: str = "beomi/Llama-3-Open-Ko-8B"
    output_dir: str = "outputs/dpo"

    # DPO specific parameters
    beta: float = 0.1
    loss_type: str = "sigmoid"  # "sigmoid", "hinge", "ipo"
    label_smoothing: float = 0.0

    # Training parameters
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # Sequence lengths
    max_length: int = 1024
    max_prompt_length: int = 512

    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
    ])

    # Quantization
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"

    # Other settings
    fp16: bool = False
    bf16: bool = True
    seed: int = 42
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DPOTrainingConfig":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        config_data = data.get("dpo", data)

        # Convert string numeric values to proper types
        if "learning_rate" in config_data and isinstance(config_data["learning_rate"], str):
            config_data["learning_rate"] = float(config_data["learning_rate"])
        if "beta" in config_data and isinstance(config_data["beta"], str):
            config_data["beta"] = float(config_data["beta"])

        return cls(**config_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "beta": self.beta,
            "loss_type": self.loss_type,
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_length": self.max_length,
            "use_lora": self.use_lora,
            "lora_r": self.lora_r,
            "use_4bit": self.use_4bit,
        }


class PreferenceDataset:
    """Dataset for DPO training with preference pairs.

    This class manages preference data for training,
    including loading, validation, and formatting.
    """

    def __init__(
        self,
        pairs: list[PreferencePair] | None = None,
    ):
        """Initialize the dataset.

        Args:
            pairs: List of preference pairs
        """
        self.pairs = pairs or []

    def add_pair(self, pair: PreferencePair) -> None:
        """Add a preference pair to the dataset."""
        self.pairs.append(pair)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PreferencePair:
        return self.pairs[idx]

    @classmethod
    def from_json(cls, path: str | Path) -> "PreferenceDataset":
        """Load dataset from JSON file.

        Expected format:
        [
            {
                "prompt": "...",
                "chosen": "...",
                "rejected": "...",
                "category": "..."  # optional
            },
            ...
        ]
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pairs = []
        for item in data:
            pairs.append(PreferencePair(
                prompt=item["prompt"],
                chosen=item["chosen"],
                rejected=item["rejected"],
                category=item.get("category", ""),
                metadata=item.get("metadata", {}),
            ))

        return cls(pairs)

    def to_json(self, path: str | Path) -> None:
        """Save dataset to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [p.to_dict() for p in self.pairs]

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def to_hf_dataset(self) -> dict[str, list[str]]:
        """Convert to Hugging Face dataset format."""
        return {
            "prompt": [p.prompt for p in self.pairs],
            "chosen": [p.chosen for p in self.pairs],
            "rejected": [p.rejected for p in self.pairs],
        }

    def split(
        self,
        eval_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple["PreferenceDataset", "PreferenceDataset"]:
        """Split dataset into train and eval sets.

        Args:
            eval_ratio: Ratio of samples for evaluation
            seed: Random seed

        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        import random
        random.seed(seed)

        pairs = self.pairs.copy()
        random.shuffle(pairs)

        split_idx = int(len(pairs) * (1 - eval_ratio))

        train_dataset = PreferenceDataset(pairs[:split_idx])
        eval_dataset = PreferenceDataset(pairs[split_idx:])

        return train_dataset, eval_dataset

    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics."""
        by_category: dict[str, int] = {}

        for pair in self.pairs:
            category = pair.category or "unknown"
            by_category[category] = by_category.get(category, 0) + 1

        return {
            "total_pairs": len(self.pairs),
            "by_category": by_category,
        }


class FinancialPreferenceGenerator:
    """Generate preference pairs from financial instruction data.

    This class converts regular instruction-output pairs into
    preference pairs for DPO training.
    """

    def __init__(self, seed: int = 42):
        """Initialize the generator.

        Args:
            seed: Random seed
        """
        import random
        self.random = random
        self.random.seed(seed)

    def generate_from_instructions(
        self,
        instructions: list[dict[str, Any]],
        strategy: str = "quality_degradation",
    ) -> list[PreferencePair]:
        """Generate preference pairs from instruction data.

        Args:
            instructions: List of instruction-output pairs
            strategy: Strategy for generating rejected responses
                - "quality_degradation": Degrade quality of original response
                - "length_variation": Create shorter/less detailed responses
                - "factual_errors": Introduce minor errors

        Returns:
            List of preference pairs
        """
        pairs = []

        for item in instructions:
            prompt = self._format_prompt(item)
            chosen = item.get("output", "")

            if strategy == "quality_degradation":
                rejected = self._degrade_quality(chosen)
            elif strategy == "length_variation":
                rejected = self._shorten_response(chosen)
            elif strategy == "factual_errors":
                rejected = self._introduce_errors(chosen)
            else:
                rejected = self._degrade_quality(chosen)

            if rejected and rejected != chosen:
                pairs.append(PreferencePair(
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    category=item.get("category", ""),
                    metadata={"strategy": strategy},
                ))

        return pairs

    def _format_prompt(self, item: dict[str, Any]) -> str:
        """Format instruction into prompt."""
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")

        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
        return f"### Instruction:\n{instruction}\n\n### Response:"

    def _degrade_quality(self, text: str) -> str:
        """Degrade response quality.

        Removes formatting, bullet points, and structure.
        """
        # Remove markdown formatting
        degraded = text.replace("**", "")
        degraded = degraded.replace("*", "")

        # Remove bullet points and make it less structured
        lines = degraded.split("\n")
        filtered_lines = []

        for line in lines:
            # Skip detailed analysis sections
            if line.strip().startswith("- ") or line.strip().startswith("• "):
                # Only keep some bullet points
                if self.random.random() < 0.5:
                    filtered_lines.append(line.replace("- ", "").replace("• ", ""))
            elif line.strip():
                filtered_lines.append(line)

        return "\n".join(filtered_lines[:len(filtered_lines) * 2 // 3])

    def _shorten_response(self, text: str) -> str:
        """Create a shorter, less detailed response."""
        lines = text.split("\n")

        # Keep only first portion
        shortened_lines = lines[:len(lines) // 3]

        return "\n".join(shortened_lines)

    def _introduce_errors(self, text: str) -> str:
        """Introduce minor factual/logical errors."""
        # Simple number manipulation
        import re

        def flip_risk_level(match: re.Match) -> str:
            level = match.group(1)
            if "높음" in level:
                return level.replace("높음", "낮음")
            elif "낮음" in level:
                return level.replace("낮음", "높음")
            return level

        # Flip risk assessments
        modified = re.sub(r"(리스크.*?)(높음|낮음)", flip_risk_level, text)

        # Flip recommendations
        if "매수" in modified:
            modified = modified.replace("매수", "매도", 1)
        elif "매도" in modified:
            modified = modified.replace("매도", "매수", 1)

        return modified


class FinancialDPOTrainer:
    """DPO Trainer for financial domain LLMs.

    This class provides a high-level interface for training
    language models using Direct Preference Optimization.
    """

    def __init__(
        self,
        config: DPOTrainingConfig,
    ):
        """Initialize the trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None

        self._training_state = {
            "is_trained": False,
            "started_at": None,
            "completed_at": None,
            "metrics": {},
        }

    def setup_model(self) -> None:
        """Setup model and tokenizer."""
        if not DPO_AVAILABLE:
            raise ImportError(
                "DPO training requires transformers, peft, and trl. "
                "Install with: pip install transformers peft trl"
            )

        # Quantization config
        bnb_config = None
        if self.config.use_4bit and BNB_AVAILABLE:
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )

        # Apply LoRA
        if self.config.use_lora:
            if self.config.use_4bit:
                self.model = prepare_model_for_kbit_training(self.model)

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.model = get_peft_model(self.model, lora_config)

    def train(
        self,
        train_dataset: PreferenceDataset,
        eval_dataset: PreferenceDataset | None = None,
    ) -> dict[str, Any]:
        """Train the model using DPO.

        Args:
            train_dataset: Training preference dataset
            eval_dataset: Optional evaluation dataset

        Returns:
            Training metrics
        """
        if self.model is None:
            self.setup_model()

        self._training_state["started_at"] = datetime.now().isoformat()

        # Convert to HF dataset format
        from datasets import Dataset as HFDataset

        train_hf = HFDataset.from_dict(train_dataset.to_hf_dataset())
        eval_hf = None
        if eval_dataset:
            eval_hf = HFDataset.from_dict(eval_dataset.to_hf_dataset())

        # Create DPO config
        dpo_config = DPOConfig(
            output_dir=self.config.output_dir,
            beta=self.config.beta,
            loss_type=self.config.loss_type,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps if eval_hf else None,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            max_length=self.config.max_length,
            max_prompt_length=self.config.max_prompt_length,
            seed=self.config.seed,
        )

        # Create trainer
        self.trainer = TRLDPOTrainer(
            model=self.model,
            ref_model=None,  # Use implicit reference model
            args=dpo_config,
            train_dataset=train_hf,
            eval_dataset=eval_hf,
            tokenizer=self.tokenizer,
        )

        # Train
        train_result = self.trainer.train()

        self._training_state["is_trained"] = True
        self._training_state["completed_at"] = datetime.now().isoformat()
        self._training_state["metrics"] = train_result.metrics

        return train_result.metrics

    def save_model(self, path: str | Path | None = None) -> None:
        """Save the trained model.

        Args:
            path: Output path (uses config.output_dir if not specified)
        """
        if self.model is None or not self._training_state["is_trained"]:
            raise ValueError("Model must be trained before saving")

        output_path = Path(path or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(output_path / "model")
        self.tokenizer.save_pretrained(output_path / "model")

        # Save training config
        config_path = output_path / "training_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, ensure_ascii=False, indent=2)

        # Save training state
        state_path = output_path / "training_state.json"
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(self._training_state, f, ensure_ascii=False, indent=2)

    def get_training_state(self) -> dict[str, Any]:
        """Get current training state."""
        return self._training_state.copy()


def create_dpo_dataset_from_instructions(
    instructions_path: str | Path,
    output_path: str | Path,
    strategy: str = "quality_degradation",
) -> PreferenceDataset:
    """Convenience function to create DPO dataset from instructions.

    Args:
        instructions_path: Path to instruction JSON file
        output_path: Path to save preference dataset
        strategy: Strategy for generating rejected responses

    Returns:
        Created PreferenceDataset
    """
    with open(instructions_path, "r", encoding="utf-8") as f:
        instructions = json.load(f)

    generator = FinancialPreferenceGenerator()
    pairs = generator.generate_from_instructions(instructions, strategy)

    dataset = PreferenceDataset(pairs)
    dataset.to_json(output_path)

    return dataset
