#!/usr/bin/env python
"""
모델 학습 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataIngestion, DataPreprocessor
from src.training import ModelTrainer

def main():
    print("=== 모델 학습 ===")

    # MLflow 설정
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    use_mlflow = mlflow_uri is not None

    # 데이터 로드
    data_path = "data/raw/transactions.csv"
    if not Path(data_path).exists():
        print("데이터 파일이 없습니다. 먼저 generate_data.py를 실행하세요.")
        # 샘플 데이터 생성
        ingestion = DataIngestion(data_dir="data")
        df = ingestion.generate_sample_data(n_samples=10000, fraud_ratio=0.02)
        ingestion.save_data(df, "raw/transactions.csv")
    else:
        ingestion = DataIngestion(data_dir="data")
        df = ingestion.load_csv(data_path)

    print(f"데이터 로드: {len(df)} samples")

    # 전처리
    preprocessor = DataPreprocessor(scaler_type="standard")
    X, y = preprocessor.prepare_features(df, include_target=True)

    # 데이터 분할
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)

    # 스케일러 학습 및 적용
    preprocessor.fit(X_train)
    X_train_scaled = preprocessor.transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)

    # 전처리기 저장
    preprocessor.save("models/preprocessor.pkl")

    # 모델 학습
    trainer = ModelTrainer(
        model_type="random_forest",
        mlflow_tracking=use_mlflow,
    )

    trainer.train(X_train_scaled, y_train, experiment_name="fraud-detection")

    # 피처 중요도
    importance = trainer.get_feature_importance(X_train.columns.tolist())
    if importance is not None:
        print("\n피처 중요도:")
        print(importance.head(10).to_string(index=False))

    # 모델 저장
    trainer.save("models/fraud_detector.pkl")

    print("\n학습 완료!")
    print(f"모델 저장: models/fraud_detector.pkl")
    print(f"전처리기 저장: models/preprocessor.pkl")

if __name__ == "__main__":
    main()
