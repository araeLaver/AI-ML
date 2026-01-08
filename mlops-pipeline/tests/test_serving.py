"""
Serving API Tests
"""

import pytest
from fastapi.testclient import TestClient

from src.serving.predictor import FraudPredictor


class TestFraudPredictor:
    """FraudPredictor 테스트"""

    def test_prepare_input(self, sample_transaction):
        """입력 준비 테스트"""
        predictor = FraudPredictor(threshold=0.5)
        df = predictor._prepare_input(sample_transaction)

        assert len(df) == 1
        assert all(f in df.columns for f in predictor.REQUIRED_FEATURES)

    def test_prepare_input_with_defaults(self):
        """기본값 적용 테스트"""
        predictor = FraudPredictor(threshold=0.5)

        minimal_input = {
            "amount": 100.0,
            "time_hour": 12,
            "location_distance": 5.0,
            "previous_avg_amount": 80.0,
        }

        df = predictor._prepare_input(minimal_input)

        # 기본값이 적용되어야 함
        assert df["day_of_week"].iloc[0] == 0
        assert df["merchant_category"].iloc[0] == 0

    def test_get_model_info(self):
        """모델 정보 테스트"""
        predictor = FraudPredictor(threshold=0.6)
        info = predictor.get_model_info()

        assert info["is_loaded"] is False
        assert info["threshold"] == 0.6
        assert "required_features" in info

    def test_update_threshold(self):
        """임계값 업데이트 테스트"""
        predictor = FraudPredictor(threshold=0.5)
        predictor.update_threshold(0.7)

        assert predictor.threshold == 0.7

    def test_update_threshold_invalid(self):
        """잘못된 임계값 테스트"""
        predictor = FraudPredictor(threshold=0.5)

        with pytest.raises(ValueError):
            predictor.update_threshold(1.5)

        with pytest.raises(ValueError):
            predictor.update_threshold(-0.1)


class TestPredictorWithModel:
    """모델이 로드된 상태의 Predictor 테스트"""

    def test_predict(self, trained_model, sample_transaction, tmp_path):
        """예측 테스트"""
        trainer, preprocessor = trained_model

        # 모델 저장
        model_path = tmp_path / "model.pkl"
        trainer.save(str(model_path))

        # Predictor 생성 및 예측
        predictor = FraudPredictor(model_path=str(model_path), threshold=0.5)
        result = predictor.predict(sample_transaction)

        assert "is_fraud" in result
        assert "probability" in result
        assert "risk_level" in result
        assert 0 <= result["probability"] <= 1

    def test_predict_batch(self, trained_model, sample_transaction, tmp_path):
        """배치 예측 테스트"""
        trainer, preprocessor = trained_model

        # 모델 저장
        model_path = tmp_path / "model.pkl"
        trainer.save(str(model_path))

        # Predictor 생성 및 배치 예측
        predictor = FraudPredictor(model_path=str(model_path), threshold=0.5)
        transactions = [sample_transaction] * 5
        results = predictor.predict_batch(transactions)

        assert len(results) == 5
        for result in results:
            assert "is_fraud" in result
            assert "probability" in result


class TestAPIEndpoints:
    """API 엔드포인트 테스트"""

    @pytest.fixture
    def client(self):
        """테스트 클라이언트"""
        from src.serving.api import app

        return TestClient(app)

    def test_root(self, client):
        """루트 엔드포인트 테스트"""
        response = client.get("/")
        assert response.status_code == 200
        assert "name" in response.json()

    def test_health(self, client):
        """헬스 체크 테스트"""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_model_info(self, client):
        """모델 정보 테스트"""
        response = client.get("/model/info")
        # 모델이 로드되지 않은 상태에서도 동작해야 함
        assert response.status_code in [200, 503]
