#!/usr/bin/env python
"""
모델 평가 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataIngestion, DataPreprocessor
from src.training import ModelTrainer, ModelEvaluator

def main():
    print("=== 모델 평가 ===")

    # 모델 로드
    model_path = "models/fraud_detector.pkl"
    preprocessor_path = "models/preprocessor.pkl"

    if not Path(model_path).exists():
        print("모델 파일이 없습니다. 먼저 train.py를 실행하세요.")
        return

    trainer = ModelTrainer.load(model_path)
    preprocessor = DataPreprocessor.load(preprocessor_path)

    # 데이터 로드
    data_path = "data/raw/transactions.csv"
    ingestion = DataIngestion(data_dir="data")

    if not Path(data_path).exists():
        df = ingestion.generate_sample_data(n_samples=10000, fraud_ratio=0.02)
    else:
        df = ingestion.load_csv(data_path)

    # 전처리
    X, y = preprocessor.prepare_features(df, include_target=True)
    _, _, X_test, _, _, y_test = preprocessor.split_data(X, y)
    X_test_scaled = preprocessor.transform(X_test)

    # 예측
    y_pred = trainer.predict(X_test_scaled)
    y_prob = trainer.predict_proba(X_test_scaled)[:, 1]

    # 평가
    evaluator = ModelEvaluator(mlflow_tracking=False)
    metrics = evaluator.evaluate(y_test.values, y_pred, y_prob)

    print("\n=== 평가 결과 ===")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    print(f"Recall:      {metrics['recall']:.4f}")
    print(f"F1 Score:    {metrics['f1_score']:.4f}")
    print(f"ROC AUC:     {metrics['roc_auc']:.4f}")
    print(f"PR AUC:      {metrics['pr_auc']:.4f}")

    # 최적 임계값 찾기
    best_threshold, best_f1 = evaluator.find_optimal_threshold(y_test.values, y_prob)
    print(f"\n최적 임계값: {best_threshold:.2f} (F1: {best_f1:.4f})")

    # 다양한 임계값 평가
    print("\n임계값별 성능:")
    threshold_results = evaluator.evaluate_at_thresholds(y_test.values, y_prob)
    print(threshold_results.to_string(index=False))

    # 리포트 저장
    evaluator.save_report("models/evaluation_report.json")
    print("\n평가 리포트 저장: models/evaluation_report.json")

if __name__ == "__main__":
    main()
