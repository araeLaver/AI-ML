# Prompt Templates
"""
프롬프트 엔지니어링 템플릿 (Step 2: Zero-shot, Few-shot, CoT)
"""

from typing import Dict, Any, Optional


# System Prompt (금융 분석 전문가)
SYSTEM_PROMPT = """당신은 10년 경력의 금융 데이터 분석가입니다.

## 역할
- 금융 거래 데이터를 분석하고 인사이트를 제공합니다
- 이상 거래를 탐지하고 설명합니다
- 리스크를 평가하고 대응 방안을 제시합니다

## 응답 규칙
1. 항상 데이터 기반으로 분석합니다
2. 불확실한 정보는 명확히 밝힙니다
3. 수치를 포함하여 구체적으로 답변합니다
4. 전문 용어는 쉽게 설명을 덧붙입니다

## 출력 형식
분석 결과는 다음 형식으로 제공합니다:
- 요약: (1-2문장)
- 상세 분석: (bullet points)
- 권장 조치: (구체적인 행동)
- 신뢰도: (높음/중간/낮음)"""


# Zero-shot 이상거래 분석 프롬프트
FRAUD_ANALYSIS_PROMPT = """다음 금융 거래가 이상 거래인지 분석해주세요.

## 거래 정보
{transaction_info}

## ML 모델 예측
- 이상거래 확률: {fraud_probability:.1%}
- 리스크 레벨: {risk_level}

## 주요 위험 요소
{risk_factors}

위 정보를 바탕으로 종합적인 분석 의견을 제시해주세요."""


# Few-shot 예시
FEW_SHOT_EXAMPLES = [
    {
        "input": """거래정보:
- 금액: 5,000만원
- 시간: 새벽 3시
- 위치: 해외
- 평소 거래금액: 50만원""",
        "output": """## 분석 결과

**요약:** 이 거래는 **높은 확률로 이상거래**로 판단됩니다.

**상세 분석:**
- 금액 이상: 거래금액(5,000만원)이 평소(50만원)의 100배
- 시간 이상: 새벽 3시는 비정상적인 거래 시간대
- 위치 이상: 해외 거래로 갑작스러운 위치 변화

**권장 조치:**
1. 즉시 거래 보류
2. 본인 확인 전화
3. 계정 임시 동결 검토

**신뢰도:** 높음 (복합적 이상 징후)"""
    },
    {
        "input": """거래정보:
- 금액: 150만원
- 시간: 오후 7시
- 위치: 평소 거주지 근처
- 평소 거래금액: 100만원""",
        "output": """## 분석 결과

**요약:** 이 거래는 **정상 거래**로 판단됩니다.

**상세 분석:**
- 금액: 평소 대비 1.5배로 합리적 범위
- 시간: 퇴근 후 일반적인 쇼핑 시간대
- 위치: 평소 활동 범위 내

**권장 조치:**
- 별도 조치 불필요
- 정상 모니터링 유지

**신뢰도:** 높음"""
    }
]


# Chain-of-Thought 프롬프트
COT_PROMPT = """다음 금융 거래를 단계별로 분석해주세요.

## 거래 정보
{transaction_info}

## 분석 단계

### 1단계: 금액 분석
- 거래 금액: {amount:,}원
- 평균 대비 비율: {amount_vs_avg}배
- 평가:

### 2단계: 시간 분석
- 거래 시간: {hour}시
- 요일: {day_of_week}
- 주말 여부: {is_weekend}
- 평가:

### 3단계: 위치 분석
- 거래 위치: {location}
- 집과의 거리: {distance}km
- 해외 거래 여부: {is_international}
- 평가:

### 4단계: 패턴 분석
- 일일 거래 횟수: {daily_count}회
- 마지막 거래 후 경과 시간: {time_since_last}분
- 평가:

### 5단계: 종합 판단
위 분석을 종합하여 이상거래 여부를 판단하고 근거를 제시해주세요.

### 최종 결론
- 판정: (이상거래/정상거래)
- 위험 수준: (높음/중간/낮음)
- 권장 조치:"""


def get_fraud_analysis_prompt(
    transaction: Dict[str, Any],
    ml_result: Optional[Dict[str, Any]] = None,
    prompt_type: str = "zero_shot",
) -> str:
    """
    거래 분석용 프롬프트 생성

    Args:
        transaction: 거래 정보 딕셔너리
        ml_result: ML 모델 예측 결과
        prompt_type: "zero_shot", "few_shot", "cot"

    Returns:
        포맷된 프롬프트 문자열
    """
    # 거래 정보 포맷팅
    transaction_info = _format_transaction(transaction)

    if prompt_type == "zero_shot":
        return _get_zero_shot_prompt(transaction_info, ml_result)
    elif prompt_type == "few_shot":
        return _get_few_shot_prompt(transaction_info, ml_result)
    elif prompt_type == "cot":
        return _get_cot_prompt(transaction, ml_result)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


def _format_transaction(transaction: Dict[str, Any]) -> str:
    """거래 정보를 텍스트로 포맷팅"""
    lines = []

    if "transaction_id" in transaction:
        lines.append(f"- 거래 ID: {transaction['transaction_id']}")
    if "amount" in transaction:
        lines.append(f"- 거래 금액: {transaction['amount']:,}원")
    if "timestamp" in transaction:
        lines.append(f"- 거래 시간: {transaction['timestamp']}")
    if "merchant_category" in transaction:
        lines.append(f"- 가맹점 유형: {transaction['merchant_category']}")
    if "location" in transaction:
        lines.append(f"- 거래 위치: {transaction['location']}")
    if "distance_from_home" in transaction:
        lines.append(f"- 집과의 거리: {transaction['distance_from_home']}km")
    if "is_international" in transaction:
        intl = "예" if transaction['is_international'] else "아니오"
        lines.append(f"- 해외 거래: {intl}")
    if "amount_vs_avg" in transaction:
        lines.append(f"- 평균 대비 금액: {transaction['amount_vs_avg']}배")
    if "daily_txn_count" in transaction:
        lines.append(f"- 일일 거래 횟수: {transaction['daily_txn_count']}회")

    return "\n".join(lines)


def _format_risk_factors(risk_factors: list) -> str:
    """위험 요소를 텍스트로 포맷팅"""
    if not risk_factors:
        return "- 특별한 위험 요소 없음"

    lines = []
    for rf in risk_factors:
        lines.append(f"- {rf['factor']}: {rf['value']} (가중치: {rf['weight']})")
    return "\n".join(lines)


def _get_zero_shot_prompt(
    transaction_info: str,
    ml_result: Optional[Dict[str, Any]] = None,
) -> str:
    """Zero-shot 프롬프트 생성"""
    fraud_probability = ml_result.get("fraud_probability", 0.5) if ml_result else 0.5
    risk_level = ml_result.get("risk_level", "unknown") if ml_result else "unknown"
    risk_factors = ml_result.get("explanation", {}).get("risk_factors", []) if ml_result else []

    return FRAUD_ANALYSIS_PROMPT.format(
        transaction_info=transaction_info,
        fraud_probability=fraud_probability,
        risk_level=risk_level,
        risk_factors=_format_risk_factors(risk_factors),
    )


def _get_few_shot_prompt(
    transaction_info: str,
    ml_result: Optional[Dict[str, Any]] = None,
) -> str:
    """Few-shot 프롬프트 생성"""
    examples_text = ""
    for i, example in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_text += f"""
### 예시 {i}
**입력:**
{example['input']}

**분석:**
{example['output']}

---
"""

    return f"""다음 예시를 참고하여 금융 거래를 분석해주세요.
{examples_text}

### 분석할 거래
{transaction_info}

위 거래에 대한 분석을 제공해주세요."""


def _get_cot_prompt(
    transaction: Dict[str, Any],
    ml_result: Optional[Dict[str, Any]] = None,
) -> str:
    """Chain-of-Thought 프롬프트 생성"""
    transaction_info = _format_transaction(transaction)

    return COT_PROMPT.format(
        transaction_info=transaction_info,
        amount=transaction.get('amount', 0),
        amount_vs_avg=transaction.get('amount_vs_avg', 1),
        hour=transaction.get('hour', 12),
        day_of_week=transaction.get('day_of_week', 0),
        is_weekend="예" if transaction.get('is_weekend', 0) else "아니오",
        location=transaction.get('location', 'unknown'),
        distance=transaction.get('distance_from_home', 0),
        is_international="예" if transaction.get('is_international', 0) else "아니오",
        daily_count=transaction.get('daily_txn_count', 1),
        time_since_last=transaction.get('time_since_last_txn', 0),
    )


if __name__ == "__main__":
    # 테스트
    sample_transaction = {
        "transaction_id": "TXN00000001",
        "amount": 5000000,
        "timestamp": "2024-06-15 03:15:00",
        "merchant_category": "해외송금",
        "location": "해외",
        "distance_from_home": 500,
        "is_international": 1,
        "amount_vs_avg": 10,
        "daily_txn_count": 5,
        "hour": 3,
        "day_of_week": 5,
        "is_weekend": 1,
        "time_since_last_txn": 10,
    }

    sample_ml_result = {
        "fraud_probability": 0.95,
        "risk_level": "critical",
        "explanation": {
            "risk_factors": [
                {"factor": "고액 거래", "value": "5,000,000원", "weight": 0.3},
                {"factor": "해외 거래", "value": "예", "weight": 0.2},
            ]
        }
    }

    print("=" * 60)
    print("Zero-shot Prompt:")
    print("=" * 60)
    print(get_fraud_analysis_prompt(sample_transaction, sample_ml_result, "zero_shot"))

    print("\n" + "=" * 60)
    print("CoT Prompt:")
    print("=" * 60)
    print(get_fraud_analysis_prompt(sample_transaction, sample_ml_result, "cot"))
