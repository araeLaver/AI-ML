"""
Great Expectations Validator Tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from src.data.ge_expectations import (
    GreatExpectationsValidator,
    ExpectationResult,
    ValidationReport,
)


class TestExpectationResult:
    """ExpectationResult 테스트"""

    def test_create_result(self):
        """결과 생성"""
        result = ExpectationResult(
            expectation_type="expect_column_to_exist",
            success=True,
            column="amount",
        )

        assert result.expectation_type == "expect_column_to_exist"
        assert result.success is True
        assert result.column == "amount"

    def test_result_with_details(self):
        """상세 정보 포함 결과"""
        result = ExpectationResult(
            expectation_type="expect_column_values_to_be_between",
            success=False,
            column="amount",
            observed_value={"min": -10, "max": 100},
            expected_value={"min": 0, "max": 1000},
            details={"below_min_count": 5},
        )

        assert result.observed_value["min"] == -10
        assert result.details["below_min_count"] == 5

    def test_result_to_dict(self):
        """결과 직렬화"""
        result = ExpectationResult(
            expectation_type="test",
            success=True,
            column="col",
            observed_value=100,
            expected_value=100,
        )

        data = result.to_dict()

        assert data["expectation_type"] == "test"
        assert data["success"] is True
        assert data["column"] == "col"


class TestValidationReport:
    """ValidationReport 테스트"""

    @pytest.fixture
    def sample_results(self):
        """샘플 결과"""
        return [
            ExpectationResult("test1", True, "col1"),
            ExpectationResult("test2", False, "col2"),
            ExpectationResult("test3", True, "col3"),
        ]

    def test_create_report(self, sample_results):
        """리포트 생성"""
        report = ValidationReport(
            suite_name="test_suite",
            success=False,
            total_expectations=3,
            successful_expectations=2,
            failed_expectations=1,
            results=sample_results,
            run_time=0.5,
        )

        assert report.suite_name == "test_suite"
        assert report.success is False
        assert report.total_expectations == 3
        assert report.successful_expectations == 2

    def test_report_to_dict(self, sample_results):
        """리포트 직렬화"""
        report = ValidationReport(
            suite_name="test",
            success=True,
            total_expectations=3,
            successful_expectations=3,
            failed_expectations=0,
            results=sample_results,
            run_time=0.1,
        )

        data = report.to_dict()

        assert data["suite_name"] == "test"
        assert data["success_rate"] == 1.0
        assert len(data["results"]) == 3

    def test_get_failures(self, sample_results):
        """실패 결과 조회"""
        report = ValidationReport(
            suite_name="test",
            success=False,
            total_expectations=3,
            successful_expectations=2,
            failed_expectations=1,
            results=sample_results,
            run_time=0.1,
        )

        failures = report.get_failures()

        assert len(failures) == 1
        assert failures[0].column == "col2"


class TestGreatExpectationsValidator:
    """GreatExpectationsValidator 테스트"""

    @pytest.fixture
    def temp_context(self):
        """임시 GX 컨텍스트"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def validator(self, temp_context):
        """검증기 인스턴스"""
        return GreatExpectationsValidator(
            context_root=temp_context,
            use_gx=False,
        )

    @pytest.fixture
    def valid_fraud_df(self):
        """유효한 사기 탐지 데이터"""
        n = 100
        return pd.DataFrame({
            "transaction_id": [f"txn_{i}" for i in range(n)],
            "amount": np.random.uniform(10, 5000, n),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "merchant_category": np.random.choice(["retail", "food", "travel"], n),
            "location": np.random.choice(["Seoul", "Busan", "Incheon"], n),
            "is_fraud": np.random.choice([0, 0, 0, 0, 1], n),  # ~20% fraud
        })

    @pytest.fixture
    def invalid_fraud_df(self):
        """유효하지 않은 사기 탐지 데이터"""
        return pd.DataFrame({
            "transaction_id": ["txn_1", "txn_1", "txn_3"],  # 중복
            "amount": [-100, 200, 200000],  # 범위 초과
            "is_fraud": [0, 2, 1],  # 잘못된 값
        })

    def test_init_creates_directory(self, temp_context):
        """초기화 시 디렉토리 생성"""
        validator = GreatExpectationsValidator(
            context_root=temp_context,
            use_gx=False,
        )
        assert Path(temp_context).exists()

    def test_create_fraud_detection_suite(self, validator):
        """사기 탐지 스위트 생성"""
        validator.create_fraud_detection_suite()

        assert "fraud_detection" in validator.list_suites()

        suite = validator.get_suite("fraud_detection")
        assert len(suite) > 0

    def test_validate_valid_data(self, validator, valid_fraud_df):
        """유효한 데이터 검증"""
        validator.create_fraud_detection_suite()

        report = validator.validate(valid_fraud_df, "fraud_detection")

        assert report.suite_name == "fraud_detection"
        assert report.total_expectations > 0
        # 대부분의 기대치가 통과해야 함
        assert report.successful_expectations >= report.total_expectations * 0.8

    def test_validate_column_exists(self, validator):
        """컬럼 존재 검증"""
        expectations = [
            {"type": "expect_column_to_exist", "column": "amount"},
            {"type": "expect_column_to_exist", "column": "missing"},
        ]
        validator.create_expectation_suite("test", expectations)

        df = pd.DataFrame({"amount": [100, 200]})
        report = validator.validate(df, "test")

        assert report.successful_expectations == 1
        assert report.failed_expectations == 1

    def test_validate_column_type(self, validator):
        """컬럼 타입 검증"""
        expectations = [
            {
                "type": "expect_column_values_to_be_of_type",
                "column": "amount",
                "expected_type": "float64",
            },
        ]
        validator.create_expectation_suite("test", expectations)

        df = pd.DataFrame({"amount": [1.0, 2.0, 3.0]})
        report = validator.validate(df, "test")

        assert report.success is True

    def test_validate_not_null(self, validator):
        """Null 검증"""
        expectations = [
            {"type": "expect_column_values_to_not_be_null", "column": "amount"},
        ]
        validator.create_expectation_suite("test", expectations)

        # 유효한 데이터
        df = pd.DataFrame({"amount": [100, 200, 300]})
        report = validator.validate(df, "test")
        assert report.success is True

        # Null 포함 데이터
        df_with_null = pd.DataFrame({"amount": [100, None, 300]})
        report2 = validator.validate(df_with_null, "test")
        assert report2.success is False

    def test_validate_unique(self, validator):
        """고유값 검증"""
        expectations = [
            {"type": "expect_column_values_to_be_unique", "column": "id"},
        ]
        validator.create_expectation_suite("test", expectations)

        # 고유한 데이터
        df = pd.DataFrame({"id": [1, 2, 3]})
        report = validator.validate(df, "test")
        assert report.success is True

        # 중복 데이터
        df_dup = pd.DataFrame({"id": [1, 1, 3]})
        report2 = validator.validate(df_dup, "test")
        assert report2.success is False

    def test_validate_between(self, validator):
        """범위 검증"""
        expectations = [
            {
                "type": "expect_column_values_to_be_between",
                "column": "amount",
                "min_value": 0,
                "max_value": 1000,
            },
        ]
        validator.create_expectation_suite("test", expectations)

        # 범위 내 데이터
        df = pd.DataFrame({"amount": [100, 500, 999]})
        report = validator.validate(df, "test")
        assert report.success is True

        # 범위 초과 데이터
        df_exceed = pd.DataFrame({"amount": [-10, 500, 2000]})
        report2 = validator.validate(df_exceed, "test")
        assert report2.success is False

    def test_validate_in_set(self, validator):
        """집합 검증"""
        expectations = [
            {
                "type": "expect_column_values_to_be_in_set",
                "column": "status",
                "value_set": ["pending", "completed", "failed"],
            },
        ]
        validator.create_expectation_suite("test", expectations)

        # 유효한 값
        df = pd.DataFrame({"status": ["pending", "completed"]})
        report = validator.validate(df, "test")
        assert report.success is True

        # 잘못된 값
        df_invalid = pd.DataFrame({"status": ["pending", "unknown"]})
        report2 = validator.validate(df_invalid, "test")
        assert report2.success is False

    def test_validate_mean_between(self, validator):
        """평균 범위 검증"""
        expectations = [
            {
                "type": "expect_column_mean_to_be_between",
                "column": "amount",
                "min_value": 400,
                "max_value": 600,
            },
        ]
        validator.create_expectation_suite("test", expectations)

        # 평균 범위 내 데이터
        df = pd.DataFrame({"amount": [400, 500, 600]})  # mean = 500
        report = validator.validate(df, "test")
        assert report.success is True

        # 평균 범위 외 데이터
        df_high = pd.DataFrame({"amount": [900, 1000, 1100]})  # mean = 1000
        report2 = validator.validate(df_high, "test")
        assert report2.success is False

    def test_validate_proportion(self, validator):
        """비율 검증"""
        expectations = [
            {
                "type": "expect_column_proportion_of_unique_values_to_be_between",
                "column": "is_fraud",
                "min_value": 0.1,
                "max_value": 0.3,
            },
        ]
        validator.create_expectation_suite("test", expectations)

        # 20% fraud (비율 내)
        df = pd.DataFrame({"is_fraud": [0, 0, 0, 0, 1]})
        report = validator.validate(df, "test")
        assert report.success is True

    def test_validate_nonexistent_suite(self, validator):
        """존재하지 않는 스위트 검증"""
        df = pd.DataFrame({"col": [1, 2, 3]})

        with pytest.raises(ValueError, match="not found"):
            validator.validate(df, "nonexistent")

    def test_validate_schema(self, validator, valid_fraud_df):
        """스키마 검증"""
        report = validator.validate_schema(valid_fraud_df)

        assert report.total_expectations > 0
        # FRAUD_SCHEMA에 정의된 컬럼 체크
        assert any(
            r.column == "transaction_id" for r in report.results
        )

    def test_validate_custom_schema(self, validator):
        """커스텀 스키마 검증"""
        schema = {
            "id": {"dtype": "int64", "nullable": False, "unique": True},
            "value": {"dtype": "float64", "min": 0, "max": 100},
        }

        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [10.0, 50.0, 90.0],
        })

        report = validator.validate_schema(df, schema)

        # 컬럼 존재 + 타입 + nullable + unique + range 체크
        assert report.total_expectations >= 6

    def test_generate_data_profile(self, validator, valid_fraud_df):
        """데이터 프로파일 생성"""
        profile = validator.generate_data_profile(valid_fraud_df, "fraud_profile")

        assert profile["name"] == "fraud_profile"
        assert profile["row_count"] == len(valid_fraud_df)
        assert "amount" in profile["columns"]
        assert "mean" in profile["columns"]["amount"]
        assert len(profile["suggested_expectations"]) > 0

    def test_profile_numeric_columns(self, validator):
        """수치형 컬럼 프로파일"""
        df = pd.DataFrame({
            "amount": [100, 200, 300, 400, 500],
        })

        profile = validator.generate_data_profile(df)

        amount_profile = profile["columns"]["amount"]
        assert "mean" in amount_profile
        assert "std" in amount_profile
        assert "min" in amount_profile
        assert "max" in amount_profile
        assert "p25" in amount_profile
        assert "p50" in amount_profile
        assert "p75" in amount_profile

    def test_profile_categorical_columns(self, validator):
        """범주형 컬럼 프로파일"""
        df = pd.DataFrame({
            "category": ["A", "B", "A", "C", "B", "A"],
        })

        profile = validator.generate_data_profile(df)

        cat_profile = profile["columns"]["category"]
        assert "top_values" in cat_profile
        assert "A" in cat_profile["top_values"]

    def test_list_suites(self, validator):
        """스위트 목록"""
        validator.create_expectation_suite("suite1", [])
        validator.create_expectation_suite("suite2", [])

        suites = validator.list_suites()

        assert "suite1" in suites
        assert "suite2" in suites

    def test_delete_suite(self, validator):
        """스위트 삭제"""
        validator.create_expectation_suite("to_delete", [])

        assert "to_delete" in validator.list_suites()

        deleted = validator.delete_suite("to_delete")

        assert deleted is True
        assert "to_delete" not in validator.list_suites()

    def test_delete_nonexistent_suite(self, validator):
        """존재하지 않는 스위트 삭제"""
        deleted = validator.delete_suite("nonexistent")
        assert deleted is False

    def test_suite_persistence(self, temp_context):
        """스위트 영속성"""
        # 첫 번째 인스턴스에서 생성
        validator1 = GreatExpectationsValidator(temp_context, use_gx=False)
        validator1.create_fraud_detection_suite()

        # 두 번째 인스턴스에서 로드
        validator2 = GreatExpectationsValidator(temp_context, use_gx=False)

        assert "fraud_detection" in validator2.list_suites()
        suite = validator2.get_suite("fraud_detection")
        assert len(suite) > 0

    def test_report_metadata(self, validator, valid_fraud_df):
        """리포트 메타데이터"""
        validator.create_fraud_detection_suite()
        report = validator.validate(valid_fraud_df, "fraud_detection")

        assert "row_count" in report.metadata
        assert "column_count" in report.metadata
        assert "columns" in report.metadata
        assert report.metadata["row_count"] == len(valid_fraud_df)

    def test_missing_column_handling(self, validator):
        """누락된 컬럼 처리"""
        expectations = [
            {
                "type": "expect_column_values_to_be_between",
                "column": "missing",
                "min_value": 0,
                "max_value": 100,
            },
        ]
        validator.create_expectation_suite("test", expectations)

        df = pd.DataFrame({"other": [1, 2, 3]})
        report = validator.validate(df, "test")

        assert report.success is False
        assert "Column not found" in str(report.results[0].details)
