#!/usr/bin/env python
"""
샘플 데이터 생성 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataIngestion, DataValidator

def main():
    print("=== 샘플 데이터 생성 ===")

    # 데이터 생성
    ingestion = DataIngestion(data_dir="data")
    df = ingestion.generate_sample_data(n_samples=10000, fraud_ratio=0.02)

    # 정보 출력
    info = ingestion.get_data_info(df)
    print(f"데이터 크기: {info['shape']}")
    print(f"Fraud ratio: {info['fraud_ratio']:.4f}")

    # 검증
    validator = DataValidator()
    result = validator.validate(df)

    if result.is_valid:
        print("✓ 데이터 검증 통과")
    else:
        print("✗ 데이터 검증 실패")
        for error in result.errors:
            print(f"  - {error}")

    if result.warnings:
        print("경고:")
        for warning in result.warnings:
            print(f"  - {warning}")

    # 저장
    output_path = ingestion.save_data(df, "raw/transactions.csv")
    print(f"데이터 저장 완료: {output_path}")

if __name__ == "__main__":
    main()
