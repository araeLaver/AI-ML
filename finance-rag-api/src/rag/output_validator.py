# -*- coding: utf-8 -*-
"""
LLM 출력 검증 모듈 (Phase 16)

응답 품질 검증, 인용 확인, 신뢰도 보정
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re
import json


class ValidationStatus(Enum):
    """검증 상태"""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass
class ValidationResult:
    """검증 결과"""
    status: ValidationStatus
    score: float  # 0.0 ~ 1.0
    checks: List[Dict[str, Any]] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status != ValidationStatus.FAILED


class OutputValidator:
    """LLM 출력 검증기"""

    # 환각 의심 패턴
    HALLUCINATION_PATTERNS = [
        r'제가 알기로는',
        r'일반적으로 알려진',
        r'보통은',
        r'아마도',
        r'제 생각에',
        r'추측하건대',
        r'~일 것입니다',
        r'~일 수 있습니다',
        r'I think',
        r'probably',
        r'maybe',
        r'perhaps',
        r'it seems',
        r'I believe',
    ]

    # 거부 응답 패턴 (정상적인 거부)
    REFUSAL_PATTERNS = [
        r'해당 정보.*없습니다',
        r'문서에.*없습니다',
        r'찾을 수 없습니다',
        r'정보가 부족합니다',
        r'제공된 문서에는',
        r'not found',
        r'no information',
        r'cannot find',
    ]

    # 위험한 조언 패턴
    RISKY_ADVICE_PATTERNS = [
        r'반드시.*투자',
        r'지금 당장.*사야',
        r'무조건.*수익',
        r'100%.*보장',
        r'절대.*손해',
        r'guaranteed',
        r'must buy',
        r'can\'t lose',
    ]

    def __init__(
        self,
        min_answer_length: int = 20,
        max_answer_length: int = 5000,
        require_source_citation: bool = True,
        confidence_threshold: float = 0.5
    ):
        self.min_answer_length = min_answer_length
        self.max_answer_length = max_answer_length
        self.require_source_citation = require_source_citation
        self.confidence_threshold = confidence_threshold

    def validate(
        self,
        answer: str,
        context_documents: List[str],
        question: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> ValidationResult:
        """
        LLM 응답 종합 검증

        Args:
            answer: LLM 생성 답변
            context_documents: 제공된 컨텍스트 문서들
            question: 원본 질문
            sources: 출처 정보 (있는 경우)

        Returns:
            ValidationResult: 검증 결과
        """
        checks = []
        issues = []
        suggestions = []
        total_score = 0.0
        check_count = 0

        # 1. 기본 길이 검증
        length_check = self._check_length(answer)
        checks.append(length_check)
        total_score += length_check["score"]
        check_count += 1
        if not length_check["passed"]:
            issues.append(length_check["message"])

        # 2. 환각 패턴 검증
        hallucination_check = self._check_hallucination(answer)
        checks.append(hallucination_check)
        total_score += hallucination_check["score"]
        check_count += 1
        if not hallucination_check["passed"]:
            issues.extend(hallucination_check.get("patterns_found", []))
            suggestions.append("문서에 기반한 객관적 표현을 사용하세요")

        # 3. 컨텍스트 기반 검증 (인용 확인)
        citation_check = self._check_citation(answer, context_documents)
        checks.append(citation_check)
        total_score += citation_check["score"]
        check_count += 1
        if not citation_check["passed"]:
            issues.append(citation_check["message"])
            suggestions.append("문서 내용을 직접 인용하세요")

        # 4. 위험한 조언 검증
        risk_check = self._check_risky_advice(answer)
        checks.append(risk_check)
        total_score += risk_check["score"]
        check_count += 1
        if not risk_check["passed"]:
            issues.extend(risk_check.get("patterns_found", []))
            suggestions.append("투자 조언이 아닌 정보 제공임을 명시하세요")

        # 5. 질문 관련성 검증
        relevance_check = self._check_relevance(answer, question)
        checks.append(relevance_check)
        total_score += relevance_check["score"]
        check_count += 1
        if not relevance_check["passed"]:
            issues.append(relevance_check["message"])

        # 6. 거부 응답 검증 (정상적인 거부인지)
        refusal_check = self._check_refusal(answer, context_documents)
        checks.append(refusal_check)
        if refusal_check.get("is_refusal"):
            # 거부 응답이 적절한지 확인
            if refusal_check["appropriate"]:
                total_score += 1.0
            else:
                total_score += 0.5
                issues.append("거부 응답이 부적절할 수 있습니다")
            check_count += 1

        # 최종 점수 계산
        final_score = total_score / check_count if check_count > 0 else 0.0

        # 상태 결정
        if final_score >= 0.8:
            status = ValidationStatus.PASSED
        elif final_score >= 0.5:
            status = ValidationStatus.WARNING
        else:
            status = ValidationStatus.FAILED

        return ValidationResult(
            status=status,
            score=round(final_score, 3),
            checks=checks,
            issues=issues,
            suggestions=suggestions
        )

    def _check_length(self, answer: str) -> Dict[str, Any]:
        """길이 검증"""
        length = len(answer)

        if length < self.min_answer_length:
            return {
                "check": "length",
                "passed": False,
                "score": 0.3,
                "message": f"답변이 너무 짧습니다 ({length}자)",
                "length": length
            }
        elif length > self.max_answer_length:
            return {
                "check": "length",
                "passed": False,
                "score": 0.7,
                "message": f"답변이 너무 깁니다 ({length}자)",
                "length": length
            }
        else:
            return {
                "check": "length",
                "passed": True,
                "score": 1.0,
                "message": "적절한 길이",
                "length": length
            }

    def _check_hallucination(self, answer: str) -> Dict[str, Any]:
        """환각 패턴 검증"""
        found_patterns = []

        for pattern in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                found_patterns.append(pattern)

        if found_patterns:
            # 패턴 수에 따른 점수 감소
            score = max(0.3, 1.0 - (len(found_patterns) * 0.15))
            return {
                "check": "hallucination",
                "passed": len(found_patterns) <= 1,
                "score": score,
                "patterns_found": found_patterns,
                "message": f"환각 의심 패턴 {len(found_patterns)}개 발견"
            }
        else:
            return {
                "check": "hallucination",
                "passed": True,
                "score": 1.0,
                "patterns_found": [],
                "message": "환각 패턴 없음"
            }

    def _check_citation(
        self,
        answer: str,
        context_documents: List[str]
    ) -> Dict[str, Any]:
        """인용 검증 - 답변이 문서에 기반하는지"""
        if not context_documents:
            return {
                "check": "citation",
                "passed": True,
                "score": 0.5,
                "message": "컨텍스트 문서 없음",
                "overlap_ratio": 0.0
            }

        # 문서에서 핵심 구문 추출
        doc_phrases = set()
        for doc in context_documents:
            # 3단어 이상의 구문 추출
            words = doc.split()
            for i in range(len(words) - 2):
                phrase = " ".join(words[i:i+3])
                if len(phrase) > 10:  # 최소 길이
                    doc_phrases.add(phrase.lower())

        # 답변에서 문서 구문 포함 여부 확인
        answer_lower = answer.lower()
        matches = sum(1 for phrase in doc_phrases if phrase in answer_lower)
        overlap_ratio = matches / len(doc_phrases) if doc_phrases else 0

        if overlap_ratio > 0.1:
            return {
                "check": "citation",
                "passed": True,
                "score": min(1.0, 0.5 + overlap_ratio),
                "message": "문서 기반 답변 확인",
                "overlap_ratio": round(overlap_ratio, 3)
            }
        else:
            return {
                "check": "citation",
                "passed": False,
                "score": 0.4,
                "message": "문서 인용이 부족합니다",
                "overlap_ratio": round(overlap_ratio, 3)
            }

    def _check_risky_advice(self, answer: str) -> Dict[str, Any]:
        """위험한 조언 검증"""
        found_patterns = []

        for pattern in self.RISKY_ADVICE_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                found_patterns.append(pattern)

        if found_patterns:
            return {
                "check": "risky_advice",
                "passed": False,
                "score": 0.2,
                "patterns_found": found_patterns,
                "message": f"위험한 조언 패턴 {len(found_patterns)}개 발견"
            }
        else:
            return {
                "check": "risky_advice",
                "passed": True,
                "score": 1.0,
                "patterns_found": [],
                "message": "위험한 조언 없음"
            }

    def _check_relevance(self, answer: str, question: str) -> Dict[str, Any]:
        """질문 관련성 검증"""
        # 질문의 핵심 키워드 추출
        question_words = set(re.findall(r'\b\w{2,}\b', question.lower()))
        answer_words = set(re.findall(r'\b\w{2,}\b', answer.lower()))

        # 공통 키워드 비율
        if not question_words:
            return {
                "check": "relevance",
                "passed": True,
                "score": 0.5,
                "message": "질문 분석 불가",
                "overlap_ratio": 0.0
            }

        common_words = question_words & answer_words
        overlap_ratio = len(common_words) / len(question_words)

        if overlap_ratio >= 0.3:
            return {
                "check": "relevance",
                "passed": True,
                "score": min(1.0, 0.6 + overlap_ratio * 0.5),
                "message": "질문과 관련된 답변",
                "overlap_ratio": round(overlap_ratio, 3)
            }
        else:
            return {
                "check": "relevance",
                "passed": False,
                "score": 0.4,
                "message": "질문과 관련성이 낮습니다",
                "overlap_ratio": round(overlap_ratio, 3)
            }

    def _check_refusal(
        self,
        answer: str,
        context_documents: List[str]
    ) -> Dict[str, Any]:
        """거부 응답 검증"""
        is_refusal = any(
            re.search(pattern, answer, re.IGNORECASE)
            for pattern in self.REFUSAL_PATTERNS
        )

        if not is_refusal:
            return {
                "check": "refusal",
                "is_refusal": False,
                "appropriate": True,
                "message": "일반 응답"
            }

        # 거부가 적절한지 확인 (컨텍스트가 충분한데 거부했는지)
        if context_documents and len(context_documents) >= 2:
            # 컨텍스트가 충분한데 거부 = 부적절할 수 있음
            return {
                "check": "refusal",
                "is_refusal": True,
                "appropriate": False,
                "message": "컨텍스트가 있지만 거부 응답"
            }
        else:
            return {
                "check": "refusal",
                "is_refusal": True,
                "appropriate": True,
                "message": "적절한 거부 응답"
            }


class ConfidenceCalibrator:
    """신뢰도 보정기"""

    def __init__(self):
        # 보정 가중치
        self.weights = {
            "source_relevance": 0.3,
            "answer_quality": 0.25,
            "citation_overlap": 0.25,
            "no_hallucination": 0.2
        }

    def calibrate(
        self,
        base_confidence: str,
        validation_result: ValidationResult,
        source_relevance_scores: List[float]
    ) -> Tuple[str, float]:
        """
        신뢰도 보정

        Args:
            base_confidence: 기본 신뢰도 (high/medium/low)
            validation_result: 검증 결과
            source_relevance_scores: 출처별 관련도 점수

        Returns:
            (보정된 신뢰도, 수치 점수)
        """
        # 기본 점수 변환
        base_score = {"high": 0.8, "medium": 0.5, "low": 0.2}.get(base_confidence, 0.5)

        # 출처 관련도 점수
        avg_relevance = (
            sum(source_relevance_scores) / len(source_relevance_scores)
            if source_relevance_scores else 0.5
        )

        # 검증 점수
        validation_score = validation_result.score

        # 인용 점수 추출
        citation_score = 0.5
        for check in validation_result.checks:
            if check.get("check") == "citation":
                citation_score = check.get("score", 0.5)
                break

        # 환각 없음 점수
        hallucination_score = 1.0
        for check in validation_result.checks:
            if check.get("check") == "hallucination":
                hallucination_score = check.get("score", 1.0)
                break

        # 가중 평균 계산
        calibrated_score = (
            self.weights["source_relevance"] * avg_relevance +
            self.weights["answer_quality"] * validation_score +
            self.weights["citation_overlap"] * citation_score +
            self.weights["no_hallucination"] * hallucination_score
        )

        # 기본 신뢰도와 보정 점수의 가중 평균
        final_score = 0.4 * base_score + 0.6 * calibrated_score

        # 신뢰도 등급 결정
        if final_score >= 0.7:
            confidence = "high"
        elif final_score >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        return confidence, round(final_score, 3)


class JSONValidator:
    """JSON 출력 검증기"""

    @staticmethod
    def validate_json(text: str, schema: Optional[Dict] = None) -> Dict[str, Any]:
        """
        JSON 형식 검증

        Args:
            text: LLM 출력 텍스트
            schema: 예상 스키마 (선택)

        Returns:
            검증 결과
        """
        # JSON 블록 추출
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 전체가 JSON인 경우
            json_str = text.strip()

        try:
            parsed = json.loads(json_str)

            if schema:
                # 스키마 검증 (간단한 필드 체크)
                missing_fields = []
                for key, expected_type in schema.items():
                    if key not in parsed:
                        missing_fields.append(key)
                    elif expected_type and not isinstance(parsed[key], expected_type):
                        missing_fields.append(f"{key} (wrong type)")

                if missing_fields:
                    return {
                        "valid": False,
                        "parsed": parsed,
                        "error": f"Missing or invalid fields: {missing_fields}"
                    }

            return {
                "valid": True,
                "parsed": parsed,
                "error": None
            }

        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "parsed": None,
                "error": str(e)
            }


def validate_rag_response(
    answer: str,
    context_documents: List[str],
    question: str,
    sources: Optional[List[Dict]] = None,
    base_confidence: str = "medium"
) -> Dict[str, Any]:
    """
    RAG 응답 종합 검증 헬퍼

    Returns:
        {
            "is_valid": bool,
            "status": str,
            "validation_score": float,
            "calibrated_confidence": str,
            "confidence_score": float,
            "issues": List[str],
            "suggestions": List[str]
        }
    """
    validator = OutputValidator()
    calibrator = ConfidenceCalibrator()

    # 검증 실행
    result = validator.validate(answer, context_documents, question, sources)

    # 신뢰도 보정
    source_scores = [s.get("relevance_score", 0.5) for s in (sources or [])]
    calibrated_confidence, confidence_score = calibrator.calibrate(
        base_confidence, result, source_scores
    )

    return {
        "is_valid": result.is_valid,
        "status": result.status.value,
        "validation_score": result.score,
        "calibrated_confidence": calibrated_confidence,
        "confidence_score": confidence_score,
        "issues": result.issues,
        "suggestions": result.suggestions,
        "checks": result.checks
    }
