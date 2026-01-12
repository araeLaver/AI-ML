# -*- coding: utf-8 -*-
"""
실시간 금융 데이터 모듈

[기능]
- 주식 시세 조회 (yfinance)
- 환율 조회
- 주요 지수 조회
- 데이터 캐싱 (5분)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class StockQuote:
    """주식 시세 데이터"""
    symbol: str
    name: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    high_52week: Optional[float] = None
    low_52week: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExchangeRate:
    """환율 데이터"""
    from_currency: str
    to_currency: str
    rate: float
    change: float
    change_percent: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketIndex:
    """시장 지수 데이터"""
    symbol: str
    name: str
    value: float
    change: float
    change_percent: float
    timestamp: datetime = field(default_factory=datetime.now)


class RealtimeDataService:
    """실시간 금융 데이터 서비스"""

    # 한국 주요 종목 매핑 (야후 파이낸스 심볼)
    KOREAN_STOCKS = {
        "삼성전자": "005930.KS",
        "SK하이닉스": "000660.KS",
        "LG에너지솔루션": "373220.KS",
        "삼성바이오로직스": "207940.KS",
        "현대차": "005380.KS",
        "기아": "000270.KS",
        "셀트리온": "068270.KS",
        "POSCO홀딩스": "005490.KS",
        "KB금융": "105560.KS",
        "신한지주": "055550.KS",
        "NAVER": "035420.KS",
        "카카오": "035720.KS",
        "LG화학": "051910.KS",
        "삼성SDI": "006400.KS",
        "현대모비스": "012330.KS",
    }

    # 주요 지수
    MARKET_INDICES = {
        "KOSPI": "^KS11",
        "KOSDAQ": "^KQ11",
        "S&P500": "^GSPC",
        "NASDAQ": "^IXIC",
        "DOW": "^DJI",
        "NIKKEI": "^N225",
        "상해종합": "000001.SS",
    }

    # 환율 쌍
    CURRENCY_PAIRS = {
        "USD/KRW": "KRW=X",
        "EUR/KRW": "EURKRW=X",
        "JPY/KRW": "JPYKRW=X",
        "CNY/KRW": "CNYKRW=X",
        "EUR/USD": "EURUSD=X",
    }

    def __init__(self, cache_ttl: int = 300):
        """
        Args:
            cache_ttl: 캐시 유효 시간 (초), 기본 5분
        """
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._yf_available = self._check_yfinance()

    def _check_yfinance(self) -> bool:
        """yfinance 사용 가능 여부 확인"""
        try:
            import yfinance
            return True
        except ImportError:
            logger.warning("yfinance not installed. Using sample data.")
            return False

    def _is_cache_valid(self, key: str) -> bool:
        """캐시 유효성 확인"""
        if key not in self._cache_timestamps:
            return False
        elapsed = (datetime.now() - self._cache_timestamps[key]).total_seconds()
        return elapsed < self.cache_ttl

    def _set_cache(self, key: str, value: Any):
        """캐시 설정"""
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.now()

    def _get_cache(self, key: str) -> Optional[Any]:
        """캐시 조회"""
        if self._is_cache_valid(key):
            return self._cache.get(key)
        return None

    def get_stock_quote(self, symbol_or_name: str) -> Optional[StockQuote]:
        """
        주식 시세 조회

        Args:
            symbol_or_name: 종목코드 또는 종목명 (예: "005930.KS" 또는 "삼성전자")
        """
        # 한글 종목명이면 심볼로 변환
        symbol = self.KOREAN_STOCKS.get(symbol_or_name, symbol_or_name)

        cache_key = f"stock_{symbol}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        if not self._yf_available:
            sample = self._get_sample_stock(symbol_or_name, symbol)
            self._set_cache(cache_key, sample)
            return sample

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="2d")

            if hist.empty:
                return self._get_sample_stock(symbol_or_name, symbol)

            current_price = hist['Close'].iloc[-1]
            prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change = current_price - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0

            quote = StockQuote(
                symbol=symbol,
                name=info.get('shortName', info.get('longName', symbol_or_name)),
                price=round(current_price, 2),
                change=round(change, 2),
                change_percent=round(change_pct, 2),
                volume=int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                dividend_yield=info.get('dividendYield'),
                high_52week=info.get('fiftyTwoWeekHigh'),
                low_52week=info.get('fiftyTwoWeekLow'),
            )

            self._set_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Error fetching stock {symbol}: {e}")
            return self._get_sample_stock(symbol_or_name, symbol)

    def _get_sample_stock(self, name: str, symbol: str) -> StockQuote:
        """샘플 주식 데이터 반환"""
        sample_data = {
            "삼성전자": StockQuote(
                symbol="005930.KS", name="삼성전자", price=71500,
                change=1200, change_percent=1.71, volume=12500000,
                market_cap=427000000000000, pe_ratio=13.5
            ),
            "SK하이닉스": StockQuote(
                symbol="000660.KS", name="SK하이닉스", price=178000,
                change=3500, change_percent=2.01, volume=3200000,
                market_cap=129000000000000, pe_ratio=8.2
            ),
            "NAVER": StockQuote(
                symbol="035420.KS", name="NAVER", price=185000,
                change=-2000, change_percent=-1.07, volume=580000,
                market_cap=30000000000000, pe_ratio=22.5
            ),
            "카카오": StockQuote(
                symbol="035720.KS", name="카카오", price=42000,
                change=-500, change_percent=-1.18, volume=2100000,
                market_cap=18600000000000, pe_ratio=35.2
            ),
        }
        return sample_data.get(name, StockQuote(
            symbol=symbol, name=name, price=50000,
            change=500, change_percent=1.0, volume=1000000
        ))

    def get_exchange_rate(self, pair: str = "USD/KRW") -> Optional[ExchangeRate]:
        """
        환율 조회

        Args:
            pair: 환율 쌍 (예: "USD/KRW")
        """
        cache_key = f"fx_{pair}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        symbol = self.CURRENCY_PAIRS.get(pair)
        if not symbol:
            return None

        if not self._yf_available:
            return self._get_sample_exchange_rate(pair)

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")

            if hist.empty:
                return self._get_sample_exchange_rate(pair)

            current = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
            change = current - prev
            change_pct = (change / prev * 100) if prev else 0

            parts = pair.split("/")
            rate = ExchangeRate(
                from_currency=parts[0],
                to_currency=parts[1],
                rate=round(current, 2),
                change=round(change, 2),
                change_percent=round(change_pct, 2),
            )

            self._set_cache(cache_key, rate)
            return rate

        except Exception as e:
            logger.error(f"Error fetching exchange rate {pair}: {e}")
            return self._get_sample_exchange_rate(pair)

    def _get_sample_exchange_rate(self, pair: str) -> ExchangeRate:
        """샘플 환율 데이터"""
        sample_data = {
            "USD/KRW": ExchangeRate("USD", "KRW", 1385.50, 5.30, 0.38),
            "EUR/KRW": ExchangeRate("EUR", "KRW", 1520.25, -3.20, -0.21),
            "JPY/KRW": ExchangeRate("JPY", "KRW", 9.15, 0.02, 0.22),
            "CNY/KRW": ExchangeRate("CNY", "KRW", 192.30, 0.80, 0.42),
        }
        parts = pair.split("/")
        return sample_data.get(pair, ExchangeRate(parts[0], parts[1], 1.0, 0, 0))

    def get_market_index(self, index_name: str) -> Optional[MarketIndex]:
        """
        시장 지수 조회

        Args:
            index_name: 지수명 (예: "KOSPI", "S&P500")
        """
        cache_key = f"index_{index_name}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        symbol = self.MARKET_INDICES.get(index_name)
        if not symbol:
            return None

        if not self._yf_available:
            return self._get_sample_index(index_name)

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")

            if hist.empty:
                return self._get_sample_index(index_name)

            current = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
            change = current - prev
            change_pct = (change / prev * 100) if prev else 0

            idx = MarketIndex(
                symbol=symbol,
                name=index_name,
                value=round(current, 2),
                change=round(change, 2),
                change_percent=round(change_pct, 2),
            )

            self._set_cache(cache_key, idx)
            return idx

        except Exception as e:
            logger.error(f"Error fetching index {index_name}: {e}")
            return self._get_sample_index(index_name)

    def _get_sample_index(self, name: str) -> MarketIndex:
        """샘플 지수 데이터"""
        sample_data = {
            "KOSPI": MarketIndex("^KS11", "KOSPI", 2520.35, 15.20, 0.61),
            "KOSDAQ": MarketIndex("^KQ11", "KOSDAQ", 735.80, -3.45, -0.47),
            "S&P500": MarketIndex("^GSPC", "S&P500", 5998.75, 28.50, 0.48),
            "NASDAQ": MarketIndex("^IXIC", "NASDAQ", 19060.45, 85.30, 0.45),
            "DOW": MarketIndex("^DJI", "DOW", 44910.20, 125.80, 0.28),
        }
        return sample_data.get(name, MarketIndex("", name, 1000, 0, 0))

    def get_stock_history(
        self,
        symbol_or_name: str,
        period: str = "1mo"
    ) -> Optional[Dict[str, Any]]:
        """
        주가 히스토리 조회 (차트용)

        Args:
            symbol_or_name: 종목코드 또는 종목명
            period: 기간 (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)

        Returns:
            dict with dates, prices, volumes
        """
        symbol = self.KOREAN_STOCKS.get(symbol_or_name, symbol_or_name)

        cache_key = f"history_{symbol}_{period}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        if not self._yf_available:
            return self._get_sample_history(symbol_or_name, period)

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            if hist.empty:
                return self._get_sample_history(symbol_or_name, period)

            data = {
                "symbol": symbol,
                "name": symbol_or_name,
                "period": period,
                "dates": [d.strftime("%Y-%m-%d") for d in hist.index],
                "open": hist['Open'].round(2).tolist(),
                "high": hist['High'].round(2).tolist(),
                "low": hist['Low'].round(2).tolist(),
                "close": hist['Close'].round(2).tolist(),
                "volume": hist['Volume'].astype(int).tolist(),
            }

            self._set_cache(cache_key, data)
            return data

        except Exception as e:
            logger.error(f"Error fetching history for {symbol}: {e}")
            return self._get_sample_history(symbol_or_name, period)

    def _get_sample_history(self, name: str, period: str) -> Dict[str, Any]:
        """샘플 히스토리 데이터 생성"""
        import random

        # 기간에 따른 데이터 포인트 수
        period_days = {
            "1d": 1, "5d": 5, "1mo": 22, "3mo": 66,
            "6mo": 132, "1y": 252, "2y": 504, "5y": 1260
        }
        days = period_days.get(period, 22)

        base_price = 70000 if "삼성" in name else 50000
        dates = []
        prices = {"open": [], "high": [], "low": [], "close": []}
        volumes = []

        current_date = datetime.now()
        price = base_price

        for i in range(days):
            date = current_date - timedelta(days=days-i)
            dates.append(date.strftime("%Y-%m-%d"))

            # 랜덤 가격 변동
            change = random.uniform(-0.03, 0.03)
            open_p = price
            close_p = price * (1 + change)
            high_p = max(open_p, close_p) * (1 + random.uniform(0, 0.01))
            low_p = min(open_p, close_p) * (1 - random.uniform(0, 0.01))

            prices["open"].append(round(open_p, 2))
            prices["high"].append(round(high_p, 2))
            prices["low"].append(round(low_p, 2))
            prices["close"].append(round(close_p, 2))
            volumes.append(random.randint(5000000, 20000000))

            price = close_p

        return {
            "symbol": self.KOREAN_STOCKS.get(name, name),
            "name": name,
            "period": period,
            "dates": dates,
            **prices,
            "volume": volumes,
        }

    def get_market_summary(self) -> Dict[str, Any]:
        """시장 요약 정보"""
        indices = {}
        for name in ["KOSPI", "KOSDAQ", "S&P500", "NASDAQ"]:
            idx = self.get_market_index(name)
            if idx:
                indices[name] = {
                    "value": idx.value,
                    "change": idx.change,
                    "change_percent": idx.change_percent,
                }

        rates = {}
        for pair in ["USD/KRW", "EUR/KRW", "JPY/KRW"]:
            rate = self.get_exchange_rate(pair)
            if rate:
                rates[pair] = {
                    "rate": rate.rate,
                    "change": rate.change,
                    "change_percent": rate.change_percent,
                }

        return {
            "timestamp": datetime.now().isoformat(),
            "indices": indices,
            "exchange_rates": rates,
        }

    def search_stock(self, query: str) -> List[Dict[str, str]]:
        """종목 검색"""
        results = []
        query_lower = query.lower()

        for name, symbol in self.KOREAN_STOCKS.items():
            if query_lower in name.lower() or query_lower in symbol.lower():
                results.append({"name": name, "symbol": symbol})

        return results[:10]


# 싱글톤 인스턴스
_realtime_service: Optional[RealtimeDataService] = None


def get_realtime_service() -> RealtimeDataService:
    """실시간 데이터 서비스 싱글톤"""
    global _realtime_service
    if _realtime_service is None:
        _realtime_service = RealtimeDataService()
    return _realtime_service
