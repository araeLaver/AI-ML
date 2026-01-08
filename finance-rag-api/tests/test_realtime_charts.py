# -*- coding: utf-8 -*-
"""
실시간 데이터 및 차트 모듈 테스트
"""

import pytest
import sys
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRealtimeDataService:
    """실시간 데이터 서비스 테스트"""

    def test_import(self):
        """모듈 임포트 테스트"""
        from src.rag.realtime_data import RealtimeDataService, get_realtime_service
        assert RealtimeDataService is not None
        assert get_realtime_service is not None

    def test_singleton_service(self):
        """싱글톤 서비스 테스트"""
        from src.rag.realtime_data import get_realtime_service
        service1 = get_realtime_service()
        service2 = get_realtime_service()
        assert service1 is service2

    def test_get_stock_quote(self):
        """주식 시세 조회 테스트"""
        from src.rag.realtime_data import get_realtime_service
        service = get_realtime_service()

        # 한글 종목명으로 조회
        quote = service.get_stock_quote("삼성전자")
        assert quote is not None
        assert quote.name == "삼성전자"
        assert quote.price > 0
        assert isinstance(quote.change_percent, float)

    def test_get_exchange_rate(self):
        """환율 조회 테스트"""
        from src.rag.realtime_data import get_realtime_service
        service = get_realtime_service()

        rate = service.get_exchange_rate("USD/KRW")
        assert rate is not None
        assert rate.from_currency == "USD"
        assert rate.to_currency == "KRW"
        assert rate.rate > 0

    def test_get_market_index(self):
        """시장 지수 조회 테스트"""
        from src.rag.realtime_data import get_realtime_service
        service = get_realtime_service()

        idx = service.get_market_index("KOSPI")
        assert idx is not None
        assert idx.name == "KOSPI"
        assert idx.value > 0

    def test_get_stock_history(self):
        """주가 히스토리 조회 테스트"""
        from src.rag.realtime_data import get_realtime_service
        service = get_realtime_service()

        history = service.get_stock_history("삼성전자", "1mo")
        assert history is not None
        assert "dates" in history
        assert "close" in history
        assert "volume" in history
        assert len(history["dates"]) > 0

    def test_get_market_summary(self):
        """시장 요약 정보 테스트"""
        from src.rag.realtime_data import get_realtime_service
        service = get_realtime_service()

        summary = service.get_market_summary()
        assert "indices" in summary
        assert "exchange_rates" in summary
        assert "timestamp" in summary
        assert "KOSPI" in summary["indices"]
        assert "USD/KRW" in summary["exchange_rates"]

    def test_search_stock(self):
        """종목 검색 테스트"""
        from src.rag.realtime_data import get_realtime_service
        service = get_realtime_service()

        results = service.search_stock("삼성")
        assert len(results) > 0
        assert any("삼성" in r["name"] for r in results)

    def test_cache(self):
        """캐시 동작 테스트"""
        from src.rag.realtime_data import get_realtime_service
        service = get_realtime_service()

        # 첫 번째 호출
        quote1 = service.get_stock_quote("SK하이닉스")

        # 두 번째 호출 (캐시에서)
        quote2 = service.get_stock_quote("SK하이닉스")

        # 같은 값이어야 함 (캐시에서 가져왔으므로)
        assert quote1.price == quote2.price
        assert quote1.timestamp == quote2.timestamp


class TestFinanceChartBuilder:
    """차트 빌더 테스트"""

    def test_import(self):
        """모듈 임포트 테스트"""
        from src.rag.charts import FinanceChartBuilder
        assert FinanceChartBuilder is not None

    def test_create_candlestick_chart(self):
        """캔들스틱 차트 생성 테스트"""
        from src.rag.charts import FinanceChartBuilder
        from src.rag.realtime_data import get_realtime_service

        service = get_realtime_service()
        history = service.get_stock_history("삼성전자", "1mo")

        fig = FinanceChartBuilder.create_candlestick_chart(history)
        assert fig is not None
        assert len(fig.data) > 0  # 데이터가 추가되었는지

    def test_create_line_chart(self):
        """라인 차트 생성 테스트"""
        from src.rag.charts import FinanceChartBuilder
        from src.rag.realtime_data import get_realtime_service

        service = get_realtime_service()
        history = service.get_stock_history("NAVER", "1mo")

        fig = FinanceChartBuilder.create_line_chart(history)
        assert fig is not None
        assert len(fig.data) > 0

    def test_create_comparison_chart(self):
        """비교 차트 생성 테스트"""
        from src.rag.charts import FinanceChartBuilder
        from src.rag.realtime_data import get_realtime_service

        service = get_realtime_service()
        datasets = [
            service.get_stock_history("삼성전자", "1mo"),
            service.get_stock_history("SK하이닉스", "1mo"),
        ]

        fig = FinanceChartBuilder.create_comparison_chart(datasets)
        assert fig is not None
        assert len(fig.data) == 2  # 두 종목

    def test_create_market_overview_chart(self):
        """시장 개요 차트 생성 테스트"""
        from src.rag.charts import FinanceChartBuilder
        from src.rag.realtime_data import get_realtime_service

        service = get_realtime_service()
        summary = service.get_market_summary()

        fig = FinanceChartBuilder.create_market_overview_chart(summary["indices"])
        assert fig is not None
        assert len(fig.data) > 0

    def test_chart_with_volume(self):
        """거래량 포함 차트 테스트"""
        from src.rag.charts import FinanceChartBuilder
        from src.rag.realtime_data import get_realtime_service

        service = get_realtime_service()
        history = service.get_stock_history("카카오", "1mo")

        # 거래량 포함
        fig_with_vol = FinanceChartBuilder.create_candlestick_chart(
            history, show_volume=True
        )
        assert fig_with_vol is not None
        assert len(fig_with_vol.data) >= 2  # 캔들스틱 + 거래량

        # 거래량 미포함
        fig_no_vol = FinanceChartBuilder.create_candlestick_chart(
            history, show_volume=False
        )
        assert fig_no_vol is not None


class TestDataclasses:
    """데이터 클래스 테스트"""

    def test_stock_quote_dataclass(self):
        """StockQuote 데이터클래스 테스트"""
        from src.rag.realtime_data import StockQuote

        quote = StockQuote(
            symbol="005930.KS",
            name="삼성전자",
            price=70000,
            change=1000,
            change_percent=1.45,
            volume=10000000,
        )

        assert quote.symbol == "005930.KS"
        assert quote.name == "삼성전자"
        assert quote.price == 70000
        assert quote.change_percent == 1.45

    def test_exchange_rate_dataclass(self):
        """ExchangeRate 데이터클래스 테스트"""
        from src.rag.realtime_data import ExchangeRate

        rate = ExchangeRate(
            from_currency="USD",
            to_currency="KRW",
            rate=1385.50,
            change=5.30,
            change_percent=0.38,
        )

        assert rate.from_currency == "USD"
        assert rate.to_currency == "KRW"
        assert rate.rate == 1385.50

    def test_market_index_dataclass(self):
        """MarketIndex 데이터클래스 테스트"""
        from src.rag.realtime_data import MarketIndex

        idx = MarketIndex(
            symbol="^KS11",
            name="KOSPI",
            value=2520.35,
            change=15.20,
            change_percent=0.61,
        )

        assert idx.symbol == "^KS11"
        assert idx.name == "KOSPI"
        assert idx.value == 2520.35


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
