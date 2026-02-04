# -*- coding: utf-8 -*-
"""
차트 이미지 분석 모듈

차트 이미지에서 유형을 분류하고 데이터를 추출합니다.
금융 차트 (주가, 재무 그래프 등) 특화.
"""

import io
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .multimodal import ChartData


class ChartType(Enum):
    """차트 유형"""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    CANDLESTICK = "candlestick"
    AREA = "area"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    WATERFALL = "waterfall"
    UNKNOWN = "unknown"


@dataclass
class ChartAnalysisConfig:
    """차트 분석 설정

    Attributes:
        min_confidence: 최소 신뢰도
        detect_text: 텍스트 감지 여부
        extract_colors: 색상 추출 여부
        extract_data_points: 데이터 포인트 추출 시도
    """
    min_confidence: float = 0.5
    detect_text: bool = True
    extract_colors: bool = True
    extract_data_points: bool = True


@dataclass
class ChartAnalysisResult:
    """차트 분석 결과

    Attributes:
        chart_type: 차트 유형
        confidence: 신뢰도
        chart_data: 추출된 데이터
        title: 차트 제목
        labels: 레이블
        colors: 사용된 색상
        metadata: 메타데이터
    """
    chart_type: ChartType
    confidence: float
    chart_data: Optional[ChartData] = None
    title: str = ""
    labels: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """텍스트 설명으로 변환"""
        parts = [f"[{self.chart_type.value} 차트]"]

        if self.title:
            parts.append(f"제목: {self.title}")

        if self.labels:
            parts.append(f"레이블: {', '.join(self.labels)}")

        if self.chart_data and self.chart_data.values:
            parts.append(f"데이터 포인트 수: {len(self.chart_data.values)}")

        parts.append(f"신뢰도: {self.confidence:.2%}")

        return "\n".join(parts)


class ChartTypeClassifier:
    """차트 유형 분류기

    이미지 특성 기반으로 차트 유형을 분류합니다.
    """

    def __init__(self):
        """초기화"""
        # 색상 범위 (HSV)
        self.color_ranges = {
            "red": ((0, 100, 100), (10, 255, 255)),
            "blue": ((100, 100, 100), (130, 255, 255)),
            "green": ((40, 100, 100), (80, 255, 255)),
            "yellow": ((20, 100, 100), (40, 255, 255)),
        }

    def classify(self, image: Any) -> tuple[ChartType, float]:
        """차트 유형 분류

        Args:
            image: PIL Image 또는 numpy array

        Returns:
            (차트 유형, 신뢰도)
        """
        if not CV2_AVAILABLE or not PIL_AVAILABLE:
            return ChartType.UNKNOWN, 0.0

        # numpy array로 변환
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        if len(img_array.shape) < 3:
            return ChartType.UNKNOWN, 0.3

        # 특성 추출
        features = self._extract_features(img_array)

        # 규칙 기반 분류
        chart_type, confidence = self._classify_by_rules(features)

        return chart_type, confidence

    def _extract_features(self, img: np.ndarray) -> dict[str, Any]:
        """이미지 특성 추출"""
        features = {}

        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 엣지 검출
        edges = cv2.Canny(gray, 50, 150)
        features["edge_density"] = np.sum(edges > 0) / edges.size

        # 라인 검출 (Hough Transform)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=10)
        features["line_count"] = len(lines) if lines is not None else 0

        # 수평/수직 라인 비율
        if lines is not None:
            horizontal = 0
            vertical = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 15 or angle > 165:
                    horizontal += 1
                elif 75 < angle < 105:
                    vertical += 1
            features["horizontal_lines"] = horizontal
            features["vertical_lines"] = vertical
        else:
            features["horizontal_lines"] = 0
            features["vertical_lines"] = 0

        # 원 검출 (Hough Circle)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30, minRadius=10, maxRadius=200
        )
        features["circle_count"] = len(circles[0]) if circles is not None else 0

        # 색상 분포
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        features["colors"] = self._analyze_colors(hsv)

        # 사각형 영역 (막대 차트 감지용)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = 0
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            if len(approx) == 4:
                rectangles += 1
        features["rectangle_count"] = rectangles

        return features

    def _analyze_colors(self, hsv: np.ndarray) -> dict[str, float]:
        """색상 분포 분석"""
        colors = {}
        total_pixels = hsv.shape[0] * hsv.shape[1]

        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            colors[color_name] = np.sum(mask > 0) / total_pixels

        return colors

    def _classify_by_rules(self, features: dict) -> tuple[ChartType, float]:
        """규칙 기반 분류"""
        scores = {
            ChartType.BAR: 0.0,
            ChartType.LINE: 0.0,
            ChartType.PIE: 0.0,
            ChartType.CANDLESTICK: 0.0,
            ChartType.SCATTER: 0.0,
            ChartType.AREA: 0.0,
        }

        # 원형 차트: 원 감지
        if features.get("circle_count", 0) >= 1:
            scores[ChartType.PIE] += 0.7

        # 막대 차트: 수직 라인, 사각형 많음
        if features.get("vertical_lines", 0) > features.get("horizontal_lines", 0):
            scores[ChartType.BAR] += 0.4
        if features.get("rectangle_count", 0) > 3:
            scores[ChartType.BAR] += 0.3

        # 선 차트: 연속적인 라인
        if features.get("line_count", 0) > 5:
            if features.get("edge_density", 0) < 0.1:
                scores[ChartType.LINE] += 0.5

        # 캔들스틱: 빨강/초록 색상 + 수직 사각형
        colors = features.get("colors", {})
        if colors.get("red", 0) > 0.05 and colors.get("green", 0) > 0.05:
            if features.get("rectangle_count", 0) > 5:
                scores[ChartType.CANDLESTICK] += 0.6

        # 영역 차트: 채워진 영역
        if features.get("edge_density", 0) < 0.05:
            scores[ChartType.AREA] += 0.3

        # 산점도: 점들이 산재
        if features.get("circle_count", 0) > 5:
            if features.get("line_count", 0) < 3:
                scores[ChartType.SCATTER] += 0.5

        # 최고 점수 찾기
        if not scores:
            return ChartType.UNKNOWN, 0.3

        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]

        if confidence < 0.3:
            return ChartType.UNKNOWN, confidence

        return best_type, min(confidence, 0.95)


class ChartDataExtractor:
    """차트 데이터 추출기

    차트 이미지에서 데이터 포인트를 추출합니다.
    """

    def __init__(self, ocr_engine=None):
        """초기화

        Args:
            ocr_engine: OCR 엔진 (텍스트 추출용)
        """
        self._ocr_engine = ocr_engine

    @property
    def ocr_engine(self):
        """OCR 엔진 (lazy loading)"""
        if self._ocr_engine is None:
            try:
                from .ocr_pipeline import OCREngine
                self._ocr_engine = OCREngine()
            except ImportError:
                pass
        return self._ocr_engine

    def extract_data(
        self,
        image: Any,
        chart_type: ChartType,
    ) -> Optional[ChartData]:
        """차트 데이터 추출

        Args:
            image: 차트 이미지
            chart_type: 차트 유형

        Returns:
            추출된 차트 데이터
        """
        if not CV2_AVAILABLE or not PIL_AVAILABLE:
            return None

        # numpy array로 변환
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # OCR로 텍스트 추출
        title, labels, values = self._extract_text_elements(img_array)

        # 차트 유형별 데이터 추출
        if chart_type == ChartType.BAR:
            extracted_values = self._extract_bar_data(img_array)
        elif chart_type == ChartType.LINE:
            extracted_values = self._extract_line_data(img_array)
        elif chart_type == ChartType.PIE:
            extracted_values = self._extract_pie_data(img_array)
        else:
            extracted_values = values

        return ChartData(
            chart_type=chart_type.value,
            title=title,
            labels=labels,
            values=extracted_values or [],
        )

    def _extract_text_elements(
        self,
        img: np.ndarray,
    ) -> tuple[str, list[str], list[float]]:
        """텍스트 요소 추출"""
        title = ""
        labels = []
        values = []

        if self.ocr_engine is None:
            return title, labels, values

        try:
            # PIL Image로 변환
            pil_image = Image.fromarray(img)
            result = self.ocr_engine.recognize(pil_image)

            # 텍스트 분석
            lines = result.text.split('\n')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 제목 (보통 상단에 위치한 긴 텍스트)
                if not title and len(line) > 5:
                    title = line
                    continue

                # 숫자 추출
                numbers = re.findall(r'[\d,\.]+', line)
                for num in numbers:
                    try:
                        value = float(num.replace(',', ''))
                        values.append(value)
                    except ValueError:
                        pass

                # 레이블 (숫자가 아닌 텍스트)
                text_only = re.sub(r'[\d,\.\%]+', '', line).strip()
                if text_only and len(text_only) > 1:
                    labels.append(text_only)

        except Exception:
            pass

        return title, labels, values

    def _extract_bar_data(self, img: np.ndarray) -> list[float]:
        """막대 차트 데이터 추출"""
        values = []

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

            # 컨투어 찾기
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 사각형 필터링 (막대)
            bars = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0

                # 세로 막대: 높이가 너비보다 큼
                if aspect_ratio > 1.5 and h > 20:
                    bars.append((x, y, w, h))

            # X 좌표로 정렬
            bars.sort(key=lambda b: b[0])

            # 높이를 값으로 변환
            max_height = img.shape[0]
            for x, y, w, h in bars:
                normalized_value = (h / max_height) * 100
                values.append(round(normalized_value, 2))

        except Exception:
            pass

        return values

    def _extract_line_data(self, img: np.ndarray) -> list[float]:
        """선 차트 데이터 추출"""
        values = []

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # 라인 검출
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=20, maxLineGap=10)

            if lines is not None:
                # 포인트 수집
                points = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    points.append((x1, y1))
                    points.append((x2, y2))

                # X 좌표로 정렬
                points.sort(key=lambda p: p[0])

                # Y 값 추출 (이미지 좌표계 반전)
                max_height = img.shape[0]
                for x, y in points[::2]:  # 일부만 샘플링
                    normalized_value = ((max_height - y) / max_height) * 100
                    values.append(round(normalized_value, 2))

        except Exception:
            pass

        return values

    def _extract_pie_data(self, img: np.ndarray) -> list[float]:
        """파이 차트 데이터 추출"""
        values = []

        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

            # 주요 색상 영역 계산
            unique_hues = {}
            h_channel = hsv[:, :, 0]

            # 원형 마스크 적용
            center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
            radius = min(center_x, center_y) - 10
            y, x = np.ogrid[:img.shape[0], :img.shape[1]]
            mask = ((x - center_x) ** 2 + (y - center_y) ** 2) <= radius ** 2

            masked_hues = h_channel[mask]

            # 색조별 픽셀 수 계산
            for hue in range(0, 180, 15):
                count = np.sum((masked_hues >= hue) & (masked_hues < hue + 15))
                if count > 100:
                    unique_hues[hue] = count

            # 비율로 변환
            total = sum(unique_hues.values())
            if total > 0:
                for hue, count in sorted(unique_hues.items()):
                    percentage = (count / total) * 100
                    if percentage > 1:
                        values.append(round(percentage, 1))

        except Exception:
            pass

        return values


class ChartAnalyzer:
    """차트 분석기

    차트 이미지를 분류하고 데이터를 추출합니다.
    """

    def __init__(self, config: Optional[ChartAnalysisConfig] = None):
        """초기화

        Args:
            config: 분석 설정
        """
        self.config = config or ChartAnalysisConfig()
        self.classifier = ChartTypeClassifier()
        self.extractor = ChartDataExtractor()
        self._stats = {
            "charts_analyzed": 0,
            "by_type": {},
        }

    @property
    def stats(self) -> dict[str, Any]:
        """통계 정보"""
        return self._stats.copy()

    def analyze(
        self,
        image: Union[str, Path, bytes, Any],
    ) -> ChartAnalysisResult:
        """차트 분석

        Args:
            image: 이미지 (경로, 바이트, PIL Image, numpy array)

        Returns:
            분석 결과
        """
        # 이미지 로드
        img = self._load_image(image)
        if img is None:
            return ChartAnalysisResult(
                chart_type=ChartType.UNKNOWN,
                confidence=0.0,
            )

        # 차트 유형 분류
        chart_type, confidence = self.classifier.classify(img)

        # 신뢰도 체크
        if confidence < self.config.min_confidence:
            chart_type = ChartType.UNKNOWN

        # 데이터 추출
        chart_data = None
        if self.config.extract_data_points and chart_type != ChartType.UNKNOWN:
            chart_data = self.extractor.extract_data(img, chart_type)

        # 색상 추출
        colors = []
        if self.config.extract_colors:
            colors = self._extract_dominant_colors(img)

        # 통계 업데이트
        self._stats["charts_analyzed"] += 1
        type_name = chart_type.value
        self._stats["by_type"][type_name] = self._stats["by_type"].get(type_name, 0) + 1

        return ChartAnalysisResult(
            chart_type=chart_type,
            confidence=confidence,
            chart_data=chart_data,
            title=chart_data.title if chart_data else "",
            labels=chart_data.labels if chart_data else [],
            colors=colors,
            metadata={"analyzed": True},
        )

    def _load_image(self, image: Union[str, Path, bytes, Any]) -> Optional[Any]:
        """이미지 로드"""
        if not PIL_AVAILABLE:
            return None

        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                return Image.open(path)
        elif isinstance(image, bytes):
            return Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            return image

        return None

    def _extract_dominant_colors(self, img: Any) -> list[str]:
        """주요 색상 추출"""
        if not CV2_AVAILABLE:
            return []

        colors = []

        try:
            if isinstance(img, Image.Image):
                img_array = np.array(img)
            else:
                img_array = img

            # k-means로 주요 색상 추출
            pixels = img_array.reshape(-1, 3).astype(np.float32)

            # 단순화: 평균 색상 계산
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = min(5, len(pixels) // 100)
            if k < 1:
                k = 1

            _, labels, centers = cv2.kmeans(
                pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )

            # RGB to HEX
            for center in centers:
                r, g, b = [int(c) for c in center]
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                colors.append(hex_color)

        except Exception:
            pass

        return colors


# 편의 함수
def analyze_chart(
    image: Union[str, Path, bytes, Any],
) -> ChartAnalysisResult:
    """차트 분석 (편의 함수)

    Args:
        image: 차트 이미지

    Returns:
        분석 결과
    """
    analyzer = ChartAnalyzer()
    return analyzer.analyze(image)


def classify_chart_type(
    image: Union[str, Path, bytes, Any],
) -> tuple[ChartType, float]:
    """차트 유형 분류 (편의 함수)

    Args:
        image: 차트 이미지

    Returns:
        (차트 유형, 신뢰도)
    """
    classifier = ChartTypeClassifier()

    if PIL_AVAILABLE:
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        else:
            img = image
        return classifier.classify(img)

    return ChartType.UNKNOWN, 0.0
