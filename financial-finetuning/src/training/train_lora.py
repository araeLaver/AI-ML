# LoRA/QLoRA Training Pipeline
"""
금융 도메인 LLM Fine-tuning을 위한 LoRA 학습 파이프라인

Features:
- LoRA/QLoRA 기반 효율적 학습
- Early Stopping 지원
- 체크포인트 저장 및 복원
- 학습 메트릭 추적
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime

import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer

from ..data import FinancialInstructionDataset

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingMetricsCallback(TrainerCallback):
    """학습 메트릭을 추적하는 콜백"""

    def __init__(self, log_dir: str = "./outputs/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        self.start_time = None

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.start_time = datetime.now()
        logger.info(f"Training started at {self.start_time}")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs:
            log_entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                **logs,
                "timestamp": datetime.now().isoformat(),
            }
            self.metrics_history.append(log_entry)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else None

        # 메트릭 저장
        metrics_file = self.log_dir / "training_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump({
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": end_time.isoformat(),
                "duration_seconds": duration.total_seconds() if duration else None,
                "total_steps": state.global_step,
                "metrics": self.metrics_history,
            }, f, indent=2, ensure_ascii=False)

        logger.info(f"Training completed. Duration: {duration}")
        logger.info(f"Metrics saved to {metrics_file}")


def load_training_config(config_path: str = "configs/training_config.yaml") -> Dict[str, Any]:
    """YAML 설정 파일 로드"""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class FinancialLoRATrainer:
    """
    금융 도메인 LoRA/QLoRA Fine-tuning 트레이너

    Usage:
        trainer = FinancialLoRATrainer(config_path="configs/training_config.yaml")
        trainer.setup()
        trainer.train()
        trainer.save_model()
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        early_stopping_patience: int = 3,
        early_stopping_threshold: float = 0.01,
        extra_callbacks: Optional[list] = None,
    ):
        """
        Args:
            config_path: 설정 파일 경로
            config: 직접 전달할 설정 딕셔너리
            early_stopping_patience: Early stopping patience (epochs)
            early_stopping_threshold: 개선 임계값
            extra_callbacks: 추가 콜백 리스트 (예: Ray Tune 리포트 콜백)
        """
        if config is not None:
            self.config = config
        elif config_path:
            self.config = load_training_config(config_path)
        else:
            self.config = load_training_config()

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.dataset = None

        # Early stopping 설정
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.extra_callbacks = extra_callbacks or []

        # 학습 상태
        self.training_complete = False
        self.best_metric = None

    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """양자화 설정 생성"""
        quant_config = self.config.get("quantization", {})

        if not quant_config.get("enabled", False):
            return None

        compute_dtype = getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "bfloat16"))

        return BitsAndBytesConfig(
            load_in_4bit=quant_config.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
        )

    def _get_lora_config(self) -> LoraConfig:
        """LoRA 설정 생성"""
        lora_config = self.config.get("lora", {})

        return LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
            target_modules=lora_config.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
        )

    def _get_training_arguments(self) -> TrainingArguments:
        """학습 인자 생성"""
        train_config = self.config.get("training", {})
        output_dir = train_config.get("output_dir", "./outputs")

        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=train_config.get("num_train_epochs", 3),
            per_device_train_batch_size=train_config.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 4),
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 4),
            learning_rate=train_config.get("learning_rate", 2e-4),
            weight_decay=train_config.get("weight_decay", 0.01),
            warmup_ratio=train_config.get("warmup_ratio", 0.03),
            lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
            optim=train_config.get("optim", "paged_adamw_8bit"),
            fp16=train_config.get("fp16", False),
            bf16=train_config.get("bf16", True),
            logging_steps=train_config.get("logging_steps", 10),
            save_strategy=train_config.get("save_strategy", "epoch"),
            evaluation_strategy=train_config.get("eval_strategy", "epoch"),
            seed=train_config.get("seed", 42),
            report_to=train_config.get("report_to", "tensorboard"),
            push_to_hub=train_config.get("push_to_hub", False),
            hub_model_id=train_config.get("hub_model_id", None),
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            group_by_length=True,
            # Early stopping 관련 설정
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,  # 최대 3개 체크포인트 유지
        )

    def setup_model(self):
        """모델 및 토크나이저 설정"""
        model_config = self.config.get("model", {})
        model_name = model_config.get("name", "beomi/Llama-3-Open-Ko-8B")

        print(f"Loading model: {model_name}")

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # 패딩 토큰 설정
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 양자화 설정
        quantization_config = self._get_quantization_config()

        # 모델 로드
        dtype = model_config.get("dtype", "float16")
        torch_dtype = getattr(torch, dtype) if dtype else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # QLoRA: k-bit 학습을 위한 모델 준비
        if quantization_config is not None:
            self.model = prepare_model_for_kbit_training(self.model)

        # LoRA 적용
        lora_config = self._get_lora_config()
        self.model = get_peft_model(self.model, lora_config)

        # 학습 가능한 파라미터 출력
        self.model.print_trainable_parameters()

        return self

    def setup_dataset(self, data_path: Optional[str] = None):
        """데이터셋 설정"""
        data_config = self.config.get("data", {})

        self.dataset = FinancialInstructionDataset(
            data_path=data_path or data_config.get("train_file"),
            test_size=data_config.get("test_size", 0.1),
            seed=self.config.get("training", {}).get("seed", 42),
            prompt_template=data_config.get("prompt_template"),
            prompt_template_no_input=data_config.get("prompt_template_no_input"),
        )

        # 통계 출력
        stats = self.dataset.get_statistics()
        print(f"\nDataset Statistics:")
        print(f"  Train samples: {stats['train_samples']}")
        print(f"  Eval samples: {stats['eval_samples']}")
        print(f"  Categories: {list(stats['categories'].keys())}")

        return self

    def setup(self, data_path: Optional[str] = None):
        """전체 설정 (모델 + 데이터셋)"""
        self.setup_model()
        self.setup_dataset(data_path)
        return self

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        모델 학습 실행

        Args:
            resume_from_checkpoint: 체크포인트에서 재개할 경로
        """
        if self.model is None or self.dataset is None:
            raise RuntimeError("Call setup() before training")

        model_config = self.config.get("model", {})
        train_config = self.config.get("training", {})
        output_dir = train_config.get("output_dir", "./outputs")

        # 콜백 설정
        callbacks = [
            TrainingMetricsCallback(log_dir=f"{output_dir}/logs"),
            EarlyStoppingCallback(
                early_stopping_patience=self.early_stopping_patience,
                early_stopping_threshold=self.early_stopping_threshold,
            ),
        ] + list(self.extra_callbacks)

        # SFTTrainer 설정
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset.get_train_dataset(),
            eval_dataset=self.dataset.get_eval_dataset(),
            args=self._get_training_arguments(),
            max_seq_length=model_config.get("max_seq_length", 2048),
            dataset_text_field="text",
            packing=False,
            callbacks=callbacks,
        )

        logger.info("Starting training...")
        logger.info(f"  - Training samples: {len(self.dataset.get_train_dataset())}")
        logger.info(f"  - Eval samples: {len(self.dataset.get_eval_dataset())}")
        logger.info(f"  - Early stopping patience: {self.early_stopping_patience}")

        try:
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            self.training_complete = True
            self.best_metric = self.trainer.state.best_metric
            logger.info(f"Training completed. Best eval loss: {self.best_metric}")
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
            self.training_complete = False
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        return self

    def save_model(self, output_dir: Optional[str] = None) -> str:
        """
        학습된 모델 저장

        저장 내용:
        - LoRA 어댑터 가중치
        - 토크나이저
        - 학습 설정
        - 학습 상태 및 메트릭
        """
        if self.trainer is None:
            raise RuntimeError("No trained model to save")

        output_path = output_dir or self.config.get("training", {}).get("output_dir", "./outputs")
        output_path = Path(output_path)

        # LoRA 어댑터 저장
        adapter_path = output_path / "lora_adapter"
        adapter_path.mkdir(parents=True, exist_ok=True)

        self.trainer.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        # 학습 설정 저장
        config_save_path = adapter_path / "training_config.json"
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        # 학습 상태 저장
        training_state = {
            "training_complete": self.training_complete,
            "best_metric": self.best_metric,
            "base_model": self.config.get("model", {}).get("name", "unknown"),
            "saved_at": datetime.now().isoformat(),
            "total_steps": self.trainer.state.global_step if self.trainer.state else None,
            "epochs_trained": self.trainer.state.epoch if self.trainer.state else None,
        }

        state_path = adapter_path / "training_state.json"
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(training_state, f, indent=2, ensure_ascii=False)

        logger.info(f"Model saved to: {adapter_path}")
        logger.info(f"  - Adapter weights: {adapter_path}")
        logger.info(f"  - Training config: {config_save_path}")
        logger.info(f"  - Training state: {state_path}")

        return str(adapter_path)

    def push_to_hub(self, repo_id: str):
        """Hugging Face Hub에 업로드"""
        if self.trainer is None:
            raise RuntimeError("No trained model to push")

        self.trainer.push_to_hub(repo_id)
        print(f"Model pushed to: https://huggingface.co/{repo_id}")


def train_financial_model(
    config_path: str = "configs/training_config.yaml",
    data_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    금융 도메인 모델 학습 실행 함수

    Args:
        config_path: 설정 파일 경로
        data_path: 커스텀 데이터 경로
        output_dir: 출력 디렉토리

    Returns:
        저장된 모델 경로
    """
    trainer = FinancialLoRATrainer(config_path=config_path)
    trainer.setup(data_path=data_path)
    trainer.train()
    return trainer.save_model(output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Financial LoRA Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to custom training data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only setup model without training (for testing)",
    )

    args = parser.parse_args()

    if args.dry_run:
        print("Dry run mode - setting up without training")
        trainer = FinancialLoRATrainer(config_path=args.config)
        # 드라이런에서는 데이터셋만 설정
        trainer.setup_dataset(args.data)
        stats = trainer.dataset.get_statistics()
        print(f"\nReady to train with {stats['train_samples']} samples")
    else:
        output_path = train_financial_model(
            config_path=args.config,
            data_path=args.data,
            output_dir=args.output,
        )
        print(f"\nTraining complete! Model saved to: {output_path}")
