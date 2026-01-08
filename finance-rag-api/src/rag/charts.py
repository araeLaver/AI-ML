# -*- coding: utf-8 -*-
"""
금융 차트 시각화 모듈

[기능]
- 캔들스틱 차트
- 라인 차트
- 지수/환율 비교 차트
- 거래량 차트
"""

from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class FinanceChartBuilder:
    """금융 차트 빌더"""

    # 차트 테마 색상
    COLORS = {
        "up": "#26a69a",        # 상승 (초록)
        "down": "#ef5350",      # 하락 (빨강)
        "accent": "#ff4d00",    # 강조색
        "grid": "#333333",      # 그리드
        "bg": "#0a0a0a",        # 배경
        "text": "#ffffff",      # 텍스트
        "volume": "#5c6bc0",    # 거래량
    }

    @classmethod
    def create_candlestick_chart(
        cls,
        data: Dict[str, Any],
        show_volume: bool = True,
        height: int = 500
    ) -> go.Figure:
        """
        캔들스틱 차트 생성

        Args:
            data: 주가 히스토리 데이터 (dates, open, high, low, close, volume)
            show_volume: 거래량 표시 여부
            height: 차트 높이
        """
        if show_volume and "volume" in data:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
            )
        else:
            fig = go.Figure()

        # 캔들스틱
        candlestick = go.Candlestick(
            x=data["dates"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="가격",
            increasing_line_color=cls.COLORS["up"],
            decreasing_line_color=cls.COLORS["down"],
        )

        if show_volume and "volume" in data:
            fig.add_trace(candlestick, row=1, col=1)

            # 거래량 바 색상 (상승/하락 구분)
            colors = [
                cls.COLORS["up"] if c >= o else cls.COLORS["down"]
                for o, c in zip(data["open"], data["close"])
            ]

            fig.add_trace(
                go.Bar(
                    x=data["dates"],
                    y=data["volume"],
                    name="거래량",
                    marker_color=colors,
                    opacity=0.7,
                ),
                row=2, col=1
            )
        else:
            fig.add_trace(candlestick)

        # 레이아웃
        title = f"{data.get('name', '')} ({data.get('symbol', '')})"
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color=cls.COLORS["text"])),
            template="plotly_dark",
            paper_bgcolor=cls.COLORS["bg"],
            plot_bgcolor=cls.COLORS["bg"],
            height=height,
            xaxis_rangeslider_visible=False,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=30),
        )

        # X축 설정 (주말 제거)
        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])],
            gridcolor=cls.COLORS["grid"],
            showgrid=True,
        )

        fig.update_yaxes(
            gridcolor=cls.COLORS["grid"],
            showgrid=True,
        )

        return fig

    @classmethod
    def create_line_chart(
        cls,
        data: Dict[str, Any],
        show_volume: bool = False,
        height: int = 400
    ) -> go.Figure:
        """
        라인 차트 생성 (종가 기준)

        Args:
            data: 주가 히스토리 데이터
            show_volume: 거래량 표시 여부
            height: 차트 높이
        """
        if show_volume and "volume" in data:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.75, 0.25],
            )
        else:
            fig = go.Figure()

        # 라인 차트
        line = go.Scatter(
            x=data["dates"],
            y=data["close"],
            mode="lines",
            name="종가",
            line=dict(color=cls.COLORS["accent"], width=2),
            fill="tozeroy",
            fillcolor="rgba(255, 77, 0, 0.1)",
        )

        if show_volume and "volume" in data:
            fig.add_trace(line, row=1, col=1)
            fig.add_trace(
                go.Bar(
                    x=data["dates"],
                    y=data["volume"],
                    name="거래량",
                    marker_color=cls.COLORS["volume"],
                    opacity=0.6,
                ),
                row=2, col=1
            )
        else:
            fig.add_trace(line)

        title = f"{data.get('name', '')} 주가 추이"
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color=cls.COLORS["text"])),
            template="plotly_dark",
            paper_bgcolor=cls.COLORS["bg"],
            plot_bgcolor=cls.COLORS["bg"],
            height=height,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=30),
        )

        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])],
            gridcolor=cls.COLORS["grid"],
        )
        fig.update_yaxes(gridcolor=cls.COLORS["grid"])

        return fig

    @classmethod
    def create_comparison_chart(
        cls,
        datasets: List[Dict[str, Any]],
        height: int = 400
    ) -> go.Figure:
        """
        다중 종목 비교 차트 (정규화된 수익률)

        Args:
            datasets: 여러 종목의 히스토리 데이터 리스트
            height: 차트 높이
        """
        fig = go.Figure()

        colors = ["#ff4d00", "#26a69a", "#5c6bc0", "#ffc107", "#e91e63"]

        for i, data in enumerate(datasets):
            if not data or "close" not in data:
                continue

            # 첫 번째 값 기준 정규화 (수익률)
            first_price = data["close"][0]
            normalized = [(p / first_price - 1) * 100 for p in data["close"]]

            fig.add_trace(go.Scatter(
                x=data["dates"],
                y=normalized,
                mode="lines",
                name=data.get("name", f"종목 {i+1}"),
                line=dict(color=colors[i % len(colors)], width=2),
            ))

        fig.update_layout(
            title=dict(
                text="종목 비교 (수익률 %)",
                font=dict(size=16, color=cls.COLORS["text"])
            ),
            template="plotly_dark",
            paper_bgcolor=cls.COLORS["bg"],
            plot_bgcolor=cls.COLORS["bg"],
            height=height,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            margin=dict(l=50, r=50, t=50, b=30),
            yaxis_title="수익률 (%)",
        )

        fig.update_xaxes(
            rangebreaks=[dict(bounds=["sat", "mon"])],
            gridcolor=cls.COLORS["grid"],
        )
        fig.update_yaxes(gridcolor=cls.COLORS["grid"], zeroline=True)

        # 0선 강조
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color=cls.COLORS["text"],
            opacity=0.3
        )

        return fig

    @classmethod
    def create_market_overview_chart(
        cls,
        indices: Dict[str, Dict[str, float]],
        height: int = 300
    ) -> go.Figure:
        """
        시장 지수 개요 차트 (막대)

        Args:
            indices: {"KOSPI": {"value": 2500, "change_percent": 0.5}, ...}
            height: 차트 높이
        """
        names = list(indices.keys())
        changes = [indices[n].get("change_percent", 0) for n in names]
        colors = [
            cls.COLORS["up"] if c >= 0 else cls.COLORS["down"]
            for c in changes
        ]

        fig = go.Figure(data=[
            go.Bar(
                x=names,
                y=changes,
                marker_color=colors,
                text=[f"{c:+.2f}%" for c in changes],
                textposition="outside",
                textfont=dict(color=cls.COLORS["text"]),
            )
        ])

        fig.update_layout(
            title=dict(
                text="주요 지수 등락률",
                font=dict(size=14, color=cls.COLORS["text"])
            ),
            template="plotly_dark",
            paper_bgcolor=cls.COLORS["bg"],
            plot_bgcolor=cls.COLORS["bg"],
            height=height,
            showlegend=False,
            margin=dict(l=40, r=40, t=50, b=30),
            yaxis_title="등락률 (%)",
        )

        fig.update_xaxes(gridcolor=cls.COLORS["grid"])
        fig.update_yaxes(gridcolor=cls.COLORS["grid"], zeroline=True)

        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color=cls.COLORS["text"],
            opacity=0.3
        )

        return fig

    @classmethod
    def create_exchange_rate_chart(
        cls,
        rates: Dict[str, Dict[str, float]],
        height: int = 250
    ) -> go.Figure:
        """
        환율 개요 차트

        Args:
            rates: {"USD/KRW": {"rate": 1385, "change_percent": 0.3}, ...}
            height: 차트 높이
        """
        pairs = list(rates.keys())
        values = [rates[p].get("rate", 0) for p in pairs]
        changes = [rates[p].get("change_percent", 0) for p in pairs]

        fig = go.Figure()

        # 환율 값 (막대)
        fig.add_trace(go.Bar(
            x=pairs,
            y=values,
            marker_color=cls.COLORS["accent"],
            text=[f"{v:,.2f}" for v in values],
            textposition="outside",
            textfont=dict(color=cls.COLORS["text"], size=12),
            name="환율",
        ))

        fig.update_layout(
            title=dict(
                text="주요 환율",
                font=dict(size=14, color=cls.COLORS["text"])
            ),
            template="plotly_dark",
            paper_bgcolor=cls.COLORS["bg"],
            plot_bgcolor=cls.COLORS["bg"],
            height=height,
            showlegend=False,
            margin=dict(l=40, r=40, t=50, b=30),
        )

        fig.update_xaxes(gridcolor=cls.COLORS["grid"])
        fig.update_yaxes(gridcolor=cls.COLORS["grid"])

        return fig

    @classmethod
    def create_price_gauge(
        cls,
        current: float,
        low_52week: float,
        high_52week: float,
        name: str = "",
        height: int = 200
    ) -> go.Figure:
        """
        52주 가격 범위 게이지

        Args:
            current: 현재 가격
            low_52week: 52주 최저
            high_52week: 52주 최고
            name: 종목명
            height: 차트 높이
        """
        # 현재 위치 비율 계산
        if high_52week > low_52week:
            position = (current - low_52week) / (high_52week - low_52week)
        else:
            position = 0.5

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=current,
            title={"text": f"{name} 52주 가격 범위"},
            delta={"reference": (high_52week + low_52week) / 2},
            gauge={
                "axis": {"range": [low_52week, high_52week]},
                "bar": {"color": cls.COLORS["accent"]},
                "steps": [
                    {"range": [low_52week, low_52week + (high_52week-low_52week)*0.3],
                     "color": "#1a472a"},
                    {"range": [low_52week + (high_52week-low_52week)*0.3,
                              low_52week + (high_52week-low_52week)*0.7],
                     "color": "#2d5a3d"},
                    {"range": [low_52week + (high_52week-low_52week)*0.7, high_52week],
                     "color": "#3d6d4f"},
                ],
                "threshold": {
                    "line": {"color": cls.COLORS["text"], "width": 2},
                    "thickness": 0.75,
                    "value": current
                }
            },
            number={"prefix": "\u20a9", "valueformat": ",.0f"},
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=cls.COLORS["bg"],
            height=height,
            margin=dict(l=30, r=30, t=60, b=30),
        )

        return fig

    @classmethod
    def create_mini_sparkline(
        cls,
        prices: List[float],
        width: int = 120,
        height: int = 40
    ) -> go.Figure:
        """
        미니 스파크라인 차트 (대시보드용)

        Args:
            prices: 가격 리스트
            width: 너비
            height: 높이
        """
        if not prices:
            return go.Figure()

        # 상승/하락 색상
        color = cls.COLORS["up"] if prices[-1] >= prices[0] else cls.COLORS["down"]

        fig = go.Figure(go.Scatter(
            y=prices,
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="tozeroy",
            fillcolor=f"rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.2])}",
        ))

        fig.update_layout(
            width=width,
            height=height,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )

        return fig
