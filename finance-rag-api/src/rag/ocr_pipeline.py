# -*- coding: utf-8 -*-
"""
OCR 파이프라인 모듈

EasyOCR, Tesseract를 활용하여 이미지에서 텍스트를 추출합니다.
한글 문서, 스캔 PDF, 금융 문서 특화.
"""

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


@dataclass
class OCRConfig:
    """OCR 설정

    Attributes:
        languages: 인식 언어 목록
        backend: OCR 백엔드 ('easyocr', 'tesseract', 'auto')
        gpu: GPU 사용 여부 (EasyOCR)
        confidence_threshold: 신뢰도 임계값
        preprocess: 전처리 적용 여부
        dpi: PDF 변환 DPI
    """
    languages: list[str] = field(default_factory=lambda: ["ko", "en"])
    backend: str = "auto"
    gpu: bool = False
    confidence_threshold: float = 0.5
    preprocess: bool = True
    dpi: int = 200


@dataclass
class OCRResult:
    """OCR 결과

    Attributes:
        text: 추출된 텍스트
        confidence: 평균 신뢰도
        boxes: 텍스트 박스 목록 [(text, bbox, confidence), ...]
        language: 감지된 언어
        processing_time: 처리 시간 (초)
    """
    text: str
    confidence: float
    boxes: list[tuple[str, tuple, float]] = field(default_factory=list)
    language: Optional[str] = None
    processing_time: float = 0.0

    def get_text_in_region(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> str:
        """특정 영역의 텍스트 반환"""
        texts = []
        for text, bbox, conf in self.boxes:
            # bbox 중심이 영역 내에 있는지 확인
            if len(bbox) >= 4:
                bx1, by1, bx2, by2 = bbox[:4]
                center_x = (bx1 + bx2) / 2
                center_y = (by1 + by2) / 2
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    texts.append(text)
        return " ".join(texts)


class ImagePreprocessor:
    """이미지 전처리기

    OCR 정확도 향상을 위한 이미지 전처리
    """

    @staticmethod
    def preprocess(image: Any, config: OCRConfig) -> Any:
        """이미지 전처리

        Args:
            image: PIL Image 또는 numpy array
            config: OCR 설정

        Returns:
            전처리된 이미지
        """
        if not CV2_AVAILABLE:
            return image

        # PIL Image를 numpy array로 변환
        if PIL_AVAILABLE and isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image

        # 그레이스케일 변환
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # 노이즈 제거
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # 이진화 (Otsu's method)
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 기울기 보정 (선택적)
        # corrected = ImagePreprocessor._deskew(binary)

        return binary

    @staticmethod
    def _deskew(image: Any) -> Any:
        """기울기 보정"""
        if not CV2_AVAILABLE:
            return image

        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            return image

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        if abs(angle) < 0.5:  # 기울기가 거의 없으면 스킵
            return image

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated


class OCREngine:
    """OCR 엔진

    EasyOCR, Tesseract를 통합 관리
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        """초기화

        Args:
            config: OCR 설정
        """
        self.config = config or OCRConfig()
        self._easyocr_reader = None
        self._stats = {
            "images_processed": 0,
            "total_characters": 0,
            "average_confidence": 0.0,
        }

    @property
    def stats(self) -> dict[str, Any]:
        """통계 정보"""
        return self._stats.copy()

    @property
    def easyocr_reader(self):
        """EasyOCR 리더 (lazy initialization)"""
        if self._easyocr_reader is None and EASYOCR_AVAILABLE:
            self._easyocr_reader = easyocr.Reader(
                self.config.languages,
                gpu=self.config.gpu,
            )
        return self._easyocr_reader

    def recognize(
        self,
        image: Union[str, Path, bytes, Any],
    ) -> OCRResult:
        """이미지에서 텍스트 인식

        Args:
            image: 이미지 (경로, 바이트, PIL Image, numpy array)

        Returns:
            OCR 결과
        """
        import time
        start_time = time.time()

        # 이미지 로드
        img = self._load_image(image)
        if img is None:
            return OCRResult(text="", confidence=0.0)

        # 전처리
        if self.config.preprocess and CV2_AVAILABLE:
            img = ImagePreprocessor.preprocess(img, self.config)

        # OCR 실행
        backend = self._select_backend()
        if backend == "easyocr":
            result = self._recognize_easyocr(img)
        elif backend == "tesseract":
            result = self._recognize_tesseract(img)
        else:
            result = OCRResult(text="", confidence=0.0)

        # 처리 시간 기록
        result.processing_time = time.time() - start_time

        # 통계 업데이트
        self._stats["images_processed"] += 1
        self._stats["total_characters"] += len(result.text)
        if result.confidence > 0:
            old_avg = self._stats["average_confidence"]
            n = self._stats["images_processed"]
            self._stats["average_confidence"] = (old_avg * (n - 1) + result.confidence) / n

        return result

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
        elif CV2_AVAILABLE and isinstance(image, np.ndarray):
            return image

        return None

    def _select_backend(self) -> str:
        """백엔드 선택"""
        if self.config.backend != "auto":
            return self.config.backend

        # 자동 선택: EasyOCR 우선 (한글 지원 우수)
        if EASYOCR_AVAILABLE:
            return "easyocr"
        elif TESSERACT_AVAILABLE:
            return "tesseract"

        return "none"

    def _recognize_easyocr(self, image: Any) -> OCRResult:
        """EasyOCR로 인식"""
        if not EASYOCR_AVAILABLE or self.easyocr_reader is None:
            return OCRResult(text="", confidence=0.0)

        try:
            # numpy array로 변환
            if PIL_AVAILABLE and isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image

            results = self.easyocr_reader.readtext(img_array)

            texts = []
            boxes = []
            total_conf = 0.0

            for bbox, text, conf in results:
                if conf >= self.config.confidence_threshold:
                    texts.append(text)
                    # bbox를 (x1, y1, x2, y2) 형식으로 변환
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    box = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
                    boxes.append((text, box, conf))
                    total_conf += conf

            avg_conf = total_conf / len(boxes) if boxes else 0.0

            return OCRResult(
                text=" ".join(texts),
                confidence=avg_conf,
                boxes=boxes,
                language="ko" if "ko" in self.config.languages else "en",
            )

        except Exception:
            return OCRResult(text="", confidence=0.0)

    def _recognize_tesseract(self, image: Any) -> OCRResult:
        """Tesseract로 인식"""
        if not TESSERACT_AVAILABLE:
            return OCRResult(text="", confidence=0.0)

        try:
            # PIL Image로 변환
            if CV2_AVAILABLE and isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # 언어 설정
            lang = "+".join(self.config.languages)
            lang = lang.replace("ko", "kor").replace("en", "eng")

            # OCR 실행
            text = pytesseract.image_to_string(pil_image, lang=lang)

            # 상세 정보 (박스)
            try:
                data = pytesseract.image_to_data(
                    pil_image, lang=lang,
                    output_type=pytesseract.Output.DICT
                )

                boxes = []
                total_conf = 0.0
                valid_count = 0

                for i in range(len(data['text'])):
                    if data['text'][i].strip():
                        conf = float(data['conf'][i]) / 100.0 if data['conf'][i] != -1 else 0.5
                        if conf >= self.config.confidence_threshold:
                            box = (
                                data['left'][i],
                                data['top'][i],
                                data['left'][i] + data['width'][i],
                                data['top'][i] + data['height'][i],
                            )
                            boxes.append((data['text'][i], box, conf))
                            total_conf += conf
                            valid_count += 1

                avg_conf = total_conf / valid_count if valid_count > 0 else 0.5

            except Exception:
                boxes = []
                avg_conf = 0.5

            return OCRResult(
                text=text.strip(),
                confidence=avg_conf,
                boxes=boxes,
            )

        except Exception:
            return OCRResult(text="", confidence=0.0)


class PDFOCRProcessor:
    """PDF OCR 처리기

    스캔된 PDF에서 텍스트 추출
    """

    def __init__(self, config: Optional[OCRConfig] = None):
        """초기화

        Args:
            config: OCR 설정
        """
        self.config = config or OCRConfig()
        self.ocr_engine = OCREngine(config)

    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        pages: Optional[list[int]] = None,
    ) -> list[OCRResult]:
        """PDF OCR 처리

        Args:
            pdf_path: PDF 파일 경로
            pages: 처리할 페이지 번호 목록 (None이면 전체)

        Returns:
            페이지별 OCR 결과
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        results: list[OCRResult] = []

        if PYMUPDF_AVAILABLE:
            results = self._process_with_pymupdf(pdf_path, pages)
        elif PIL_AVAILABLE:
            results = self._process_with_pil(pdf_path, pages)

        return results

    def process_pdf_bytes(
        self,
        pdf_bytes: bytes,
        pages: Optional[list[int]] = None,
    ) -> list[OCRResult]:
        """PDF 바이트 OCR 처리

        Args:
            pdf_bytes: PDF 바이트
            pages: 처리할 페이지 번호 목록

        Returns:
            페이지별 OCR 결과
        """
        results: list[OCRResult] = []

        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                page_nums = pages or list(range(len(doc)))

                for page_num in page_nums:
                    if 0 <= page_num < len(doc):
                        page = doc[page_num]
                        pix = page.get_pixmap(dpi=self.config.dpi)

                        # PIL Image로 변환
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                        # OCR 실행
                        result = self.ocr_engine.recognize(img)
                        results.append(result)

                doc.close()
            except Exception:
                pass

        return results

    def _process_with_pymupdf(
        self,
        pdf_path: Path,
        pages: Optional[list[int]],
    ) -> list[OCRResult]:
        """PyMuPDF로 처리"""
        results = []

        try:
            doc = fitz.open(str(pdf_path))
            page_nums = pages or list(range(len(doc)))

            for page_num in page_nums:
                if 0 <= page_num < len(doc):
                    page = doc[page_num]

                    # 페이지를 이미지로 렌더링
                    pix = page.get_pixmap(dpi=self.config.dpi)

                    # PIL Image로 변환
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    # OCR 실행
                    result = self.ocr_engine.recognize(img)
                    results.append(result)

            doc.close()
        except Exception:
            pass

        return results

    def _process_with_pil(
        self,
        pdf_path: Path,
        pages: Optional[list[int]],
    ) -> list[OCRResult]:
        """PIL/pdf2image로 처리 (폴백)"""
        results = []

        try:
            from pdf2image import convert_from_path
            images = convert_from_path(str(pdf_path), dpi=self.config.dpi)

            page_nums = pages or list(range(len(images)))

            for page_num in page_nums:
                if 0 <= page_num < len(images):
                    result = self.ocr_engine.recognize(images[page_num])
                    results.append(result)

        except ImportError:
            pass
        except Exception:
            pass

        return results


class FinancialOCRPostProcessor:
    """금융 문서 OCR 후처리기

    금융 용어, 숫자 형식 보정
    """

    # 금융 용어 보정 맵
    CORRECTIONS = {
        "삼성잔자": "삼성전자",
        "현대차동차": "현대자동차",
        "맴출액": "매출액",
        "영엽이익": "영업이익",
        "당기슌이익": "당기순이익",
    }

    # 숫자 패턴
    NUMBER_PATTERN = re.compile(r'[\d,\.\-\(\)]+')

    @staticmethod
    def postprocess(result: OCRResult) -> OCRResult:
        """후처리 적용

        Args:
            result: OCR 결과

        Returns:
            보정된 OCR 결과
        """
        corrected_text = result.text

        # 금융 용어 보정
        for wrong, correct in FinancialOCRPostProcessor.CORRECTIONS.items():
            corrected_text = corrected_text.replace(wrong, correct)

        # 숫자 형식 보정
        corrected_text = FinancialOCRPostProcessor._fix_numbers(corrected_text)

        return OCRResult(
            text=corrected_text,
            confidence=result.confidence,
            boxes=result.boxes,
            language=result.language,
            processing_time=result.processing_time,
        )

    @staticmethod
    def _fix_numbers(text: str) -> str:
        """숫자 형식 보정"""
        # 'O'를 '0'으로, 'l'을 '1'로 변환 (숫자 컨텍스트에서)
        def replace_in_number(match):
            num = match.group(0)
            num = num.replace('O', '0').replace('o', '0')
            num = num.replace('l', '1').replace('I', '1')
            return num

        return FinancialOCRPostProcessor.NUMBER_PATTERN.sub(replace_in_number, text)


# 편의 함수
def ocr_image(
    image: Union[str, Path, bytes, Any],
    languages: list[str] = None,
) -> OCRResult:
    """이미지 OCR (편의 함수)

    Args:
        image: 이미지
        languages: 인식 언어

    Returns:
        OCR 결과
    """
    config = OCRConfig(languages=languages or ["ko", "en"])
    engine = OCREngine(config)
    return engine.recognize(image)


def ocr_pdf(
    pdf_path: Union[str, Path],
    pages: Optional[list[int]] = None,
    languages: list[str] = None,
) -> list[OCRResult]:
    """PDF OCR (편의 함수)

    Args:
        pdf_path: PDF 파일 경로
        pages: 처리할 페이지
        languages: 인식 언어

    Returns:
        페이지별 OCR 결과
    """
    config = OCRConfig(languages=languages or ["ko", "en"])
    processor = PDFOCRProcessor(config)
    return processor.process_pdf(pdf_path, pages)
