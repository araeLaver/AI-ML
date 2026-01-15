"""
금융 RAG 평가 데이터셋

100개의 평가 쿼리와 예상 정답을 포함합니다.

[데이터셋 구성]
- 실적 관련 (25개): 매출, 영업이익, 순이익 등
- 주가/시장 (20개): 주가, 시총, 거래량 등
- 산업/기술 (20개): 반도체, 배터리, AI 등
- 투자/재무 (15개): 투자, 배당, 유상증자 등
- 공시/규제 (10개): 공시, 규제, 정책 등
- 기업 정보 (10개): 기업 개요, 사업 구조 등

[정답 유형]
- exact: 정확한 수치/사실 (예: "9조 1834억원")
- relevant: 관련 정보 포함 (예: "HBM 관련 내용")
- any: 유효한 답변이면 정답

[평가 기준]
- 검색 정확도: 관련 문서 검색 여부
- 답변 정확도: 질문에 맞는 정확한 답변 여부
- 환각 방지: 없는 정보 생성 여부
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class QueryCategory(str, Enum):
    """쿼리 카테고리"""
    EARNINGS = "실적"
    MARKET = "주가/시장"
    INDUSTRY = "산업/기술"
    INVESTMENT = "투자/재무"
    DISCLOSURE = "공시/규제"
    COMPANY = "기업정보"


class AnswerType(str, Enum):
    """정답 유형"""
    EXACT = "exact"  # 정확한 수치/사실 필요
    RELEVANT = "relevant"  # 관련 정보 포함되면 OK
    ANY = "any"  # 유효한 답변이면 OK


@dataclass
class EvaluationQuery:
    """평가 쿼리"""
    id: str
    query: str
    category: QueryCategory
    answer_type: AnswerType
    expected_keywords: List[str]  # 답변에 포함되어야 할 키워드
    expected_answer: Optional[str]  # 예상 정답 (참고용)
    relevant_sources: List[str]  # 관련 문서 소스 패턴
    difficulty: str  # easy, medium, hard


# 100개 평가 쿼리
EVALUATION_QUERIES: List[EvaluationQuery] = [
    # ========================================
    # 실적 관련 (25개)
    # ========================================
    EvaluationQuery(
        id="earn_001",
        query="삼성전자 2024년 3분기 영업이익은 얼마인가요?",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.EXACT,
        expected_keywords=["영업이익", "9조", "삼성전자"],
        expected_answer="삼성전자 2024년 3분기 영업이익은 9조 1,834억원입니다.",
        relevant_sources=["삼성전자", "실적", "공시"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="earn_002",
        query="SK하이닉스 반도체 실적이 어떻게 되나요?",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["SK하이닉스", "반도체", "HBM", "실적"],
        expected_answer="SK하이닉스는 HBM 수요 증가로 2024년 사상 최대 실적이 전망됩니다.",
        relevant_sources=["SK하이닉스", "반도체"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="earn_003",
        query="현대자동차 전기차 판매량 목표",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.EXACT,
        expected_keywords=["현대자동차", "전기차", "50만대"],
        expected_answer="현대자동차는 2024년 글로벌 전기차 판매 50만대를 목표로 하고 있습니다.",
        relevant_sources=["현대자동차", "전기차"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="earn_004",
        query="카카오 광고 매출 성장률",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["카카오", "광고", "매출", "성장"],
        expected_answer="카카오는 2024년 2분기 광고 매출이 성장했습니다.",
        relevant_sources=["카카오"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="earn_005",
        query="네이버 검색광고 실적",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["네이버", "검색광고", "매출"],
        expected_answer="네이버 검색광고 매출이 증가했습니다.",
        relevant_sources=["네이버"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="earn_006",
        query="KB금융 순이익 규모",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.EXACT,
        expected_keywords=["KB금융", "순이익", "1조"],
        expected_answer="KB금융 순이익이 1조원을 돌파했습니다.",
        relevant_sources=["KB금융"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="earn_007",
        query="LG에너지솔루션 배터리 수주 현황",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["LG에너지솔루션", "배터리", "테슬라", "GM"],
        expected_answer="LG에너지솔루션은 테슬라, GM 납품 계약을 확대했습니다.",
        relevant_sources=["LG에너지솔루션"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="earn_008",
        query="삼성SDI 전고체 배터리 양산 시기",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.EXACT,
        expected_keywords=["삼성SDI", "전고체", "2027"],
        expected_answer="삼성SDI는 2027년 전고체 배터리 양산을 목표로 하고 있습니다.",
        relevant_sources=["삼성SDI"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="earn_009",
        query="포스코홀딩스 리튬 투자",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["POSCO", "리튬", "아르헨티나"],
        expected_answer="POSCO홀딩스는 아르헨티나 리튬 염호에 투자했습니다.",
        relevant_sources=["POSCO"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="earn_010",
        query="신한지주 디지털 전환 성과",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["신한", "디지털", "SOL", "2000만"],
        expected_answer="신한 쏠(SOL) 앱 가입자가 2000만을 돌파했습니다.",
        relevant_sources=["신한지주"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="earn_011",
        query="삼성전자 반도체 부문 실적 개선 요인",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["삼성전자", "반도체", "HBM", "수요"],
        expected_answer="삼성전자 반도체 부문은 HBM 수요 증가로 실적이 개선되었습니다.",
        relevant_sources=["삼성전자", "반도체"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="earn_012",
        query="현대차 아이오닉6 유럽 판매",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["현대", "아이오닉", "유럽"],
        expected_answer="현대자동차 아이오닉6가 유럽에서 판매 호조를 보이고 있습니다.",
        relevant_sources=["현대자동차"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="earn_013",
        query="카카오톡 비즈니스 플랫폼 수익성",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["카카오톡", "비즈니스", "수익"],
        expected_answer="카카오톡 비즈니스 플랫폼의 수익성이 개선되었습니다.",
        relevant_sources=["카카오"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="earn_014",
        query="KB금융 비은행 부문 성장",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["KB금융", "비은행", "다각화"],
        expected_answer="KB금융은 비은행 부문 성장으로 수익 다각화에 성공했습니다.",
        relevant_sources=["KB금융"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="earn_015",
        query="LG에너지솔루션 북미 공장 현황",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["LG에너지솔루션", "북미", "공장", "증설"],
        expected_answer="LG에너지솔루션은 북미 배터리 공장을 증설하고 있습니다.",
        relevant_sources=["LG에너지솔루션"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="earn_016",
        query="삼성바이오로직스 분기 매출",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["삼성바이오로직스", "매출"],
        expected_answer=None,
        relevant_sources=["삼성바이오로직스"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="earn_017",
        query="기아 전기차 판매 실적",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["기아", "전기차"],
        expected_answer=None,
        relevant_sources=["기아"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="earn_018",
        query="셀트리온 바이오시밀러 매출",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["셀트리온", "바이오시밀러"],
        expected_answer=None,
        relevant_sources=["셀트리온"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="earn_019",
        query="하나금융지주 이자이익",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["하나금융", "이자"],
        expected_answer=None,
        relevant_sources=["하나금융"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="earn_020",
        query="LG화학 석유화학 부문 실적",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["LG화학", "석유화학"],
        expected_answer=None,
        relevant_sources=["LG화학"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="earn_021",
        query="현대모비스 자율주행 매출",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["현대모비스", "자율주행"],
        expected_answer=None,
        relevant_sources=["현대모비스"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="earn_022",
        query="삼성물산 건설 수주",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["삼성물산", "건설", "수주"],
        expected_answer=None,
        relevant_sources=["삼성물산"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="earn_023",
        query="삼성생명 보험료 수익",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["삼성생명", "보험"],
        expected_answer=None,
        relevant_sources=["삼성생명"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="earn_024",
        query="KT&G 담배 수출 실적",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["KT&G", "담배", "수출"],
        expected_answer=None,
        relevant_sources=["KT&G"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="earn_025",
        query="SK이노베이션 배터리 흑자 전환",
        category=QueryCategory.EARNINGS,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["SK이노베이션", "배터리", "흑자"],
        expected_answer=None,
        relevant_sources=["SK이노베이션"],
        difficulty="hard"
    ),

    # ========================================
    # 주가/시장 (20개)
    # ========================================
    EvaluationQuery(
        id="mkt_001",
        query="삼성전자 현재 주가",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["삼성전자", "주가"],
        expected_answer=None,
        relevant_sources=["삼성전자", "주가"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="mkt_002",
        query="코스피 시가총액 1위 기업",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.EXACT,
        expected_keywords=["삼성전자", "시가총액", "1위"],
        expected_answer="코스피 시가총액 1위 기업은 삼성전자입니다.",
        relevant_sources=["삼성전자"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="mkt_003",
        query="SK하이닉스 PER",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["SK하이닉스", "PER"],
        expected_answer=None,
        relevant_sources=["SK하이닉스"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="mkt_004",
        query="네이버 배당 수익률",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["네이버", "배당"],
        expected_answer=None,
        relevant_sources=["네이버"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="mkt_005",
        query="카카오 52주 최고가",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["카카오", "최고가"],
        expected_answer=None,
        relevant_sources=["카카오"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="mkt_006",
        query="현대차 외국인 지분율",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["현대차", "외국인", "지분"],
        expected_answer=None,
        relevant_sources=["현대차"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="mkt_007",
        query="반도체 관련주 추천",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["반도체", "삼성전자", "SK하이닉스"],
        expected_answer="반도체 관련주로는 삼성전자, SK하이닉스가 있습니다.",
        relevant_sources=["반도체"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="mkt_008",
        query="2차전지 대장주",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["2차전지", "LG에너지솔루션", "배터리"],
        expected_answer="2차전지 대장주는 LG에너지솔루션입니다.",
        relevant_sources=["배터리", "2차전지"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="mkt_009",
        query="금융주 실적",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["금융", "KB", "신한"],
        expected_answer=None,
        relevant_sources=["금융"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="mkt_010",
        query="코스닥 AI 관련주",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["코스닥", "AI"],
        expected_answer=None,
        relevant_sources=["AI", "코스닥"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="mkt_011",
        query="삼성전자 공매도 현황",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["삼성전자", "공매도"],
        expected_answer=None,
        relevant_sources=["삼성전자", "공매도"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="mkt_012",
        query="코스피200 ETF 종류",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["코스피200", "ETF"],
        expected_answer=None,
        relevant_sources=["ETF"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="mkt_013",
        query="달러 환율과 수출주 관계",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["환율", "수출"],
        expected_answer=None,
        relevant_sources=["환율"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="mkt_014",
        query="기관 매수 종목",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["기관", "매수"],
        expected_answer=None,
        relevant_sources=["기관"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="mkt_015",
        query="MSCI 한국 편입 종목",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["MSCI", "한국"],
        expected_answer=None,
        relevant_sources=["MSCI"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="mkt_016",
        query="코스피 거래대금 상위",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["코스피", "거래대금"],
        expected_answer=None,
        relevant_sources=["거래대금"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="mkt_017",
        query="신용거래 많은 종목",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["신용거래"],
        expected_answer=None,
        relevant_sources=["신용거래"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="mkt_018",
        query="삼성전자 PBR",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["삼성전자", "PBR"],
        expected_answer=None,
        relevant_sources=["삼성전자"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="mkt_019",
        query="배당성장주 추천",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["배당", "성장"],
        expected_answer=None,
        relevant_sources=["배당"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="mkt_020",
        query="저PER 가치주",
        category=QueryCategory.MARKET,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["저PER", "가치주"],
        expected_answer=None,
        relevant_sources=["가치투자"],
        difficulty="hard"
    ),

    # ========================================
    # 산업/기술 (20개)
    # ========================================
    EvaluationQuery(
        id="ind_001",
        query="HBM 메모리란 무엇인가요?",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["HBM", "고대역폭", "메모리", "AI"],
        expected_answer="HBM(High Bandwidth Memory)은 고대역폭 메모리로, AI 가속기에 사용됩니다.",
        relevant_sources=["HBM", "반도체"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="ind_002",
        query="SK하이닉스 HBM3E 양산",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["SK하이닉스", "HBM3E", "양산"],
        expected_answer="SK하이닉스가 HBM3E 양산을 시작했습니다.",
        relevant_sources=["SK하이닉스", "HBM"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="ind_003",
        query="전고체 배터리 장점",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["전고체", "배터리", "안전", "에너지밀도"],
        expected_answer="전고체 배터리는 안전성이 높고 에너지 밀도가 우수합니다.",
        relevant_sources=["전고체", "배터리"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="ind_004",
        query="네이버 하이퍼클로바X 서비스",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["네이버", "하이퍼클로바", "AI"],
        expected_answer="네이버 AI 서비스 하이퍼클로바X가 상용화되고 있습니다.",
        relevant_sources=["네이버", "AI"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="ind_005",
        query="파운드리 사업이란",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["파운드리", "반도체", "위탁생산"],
        expected_answer="파운드리는 반도체 위탁생산 사업입니다.",
        relevant_sources=["파운드리"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="ind_006",
        query="리튬 가격 동향",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["리튬", "가격", "배터리"],
        expected_answer=None,
        relevant_sources=["리튬"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="ind_007",
        query="자율주행 기술 수준",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["자율주행", "레벨"],
        expected_answer=None,
        relevant_sources=["자율주행"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="ind_008",
        query="반도체 공정 미세화",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["반도체", "공정", "나노"],
        expected_answer=None,
        relevant_sources=["반도체"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="ind_009",
        query="OLED와 LCD 차이",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["OLED", "LCD", "디스플레이"],
        expected_answer=None,
        relevant_sources=["디스플레이"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="ind_010",
        query="수소차 vs 전기차 장단점",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["수소차", "전기차"],
        expected_answer=None,
        relevant_sources=["수소차", "전기차"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="ind_011",
        query="바이오시밀러란",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["바이오시밀러", "바이오"],
        expected_answer="바이오시밀러는 바이오의약품의 복제약입니다.",
        relevant_sources=["바이오시밀러"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="ind_012",
        query="5G와 6G 차이",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["5G", "6G", "통신"],
        expected_answer=None,
        relevant_sources=["통신"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="ind_013",
        query="클라우드 시장 경쟁",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["클라우드", "AWS", "Azure"],
        expected_answer=None,
        relevant_sources=["클라우드"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="ind_014",
        query="메타버스 관련 기술",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["메타버스", "VR", "AR"],
        expected_answer=None,
        relevant_sources=["메타버스"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="ind_015",
        query="양자컴퓨터 상용화",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["양자컴퓨터"],
        expected_answer=None,
        relevant_sources=["양자컴퓨터"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="ind_016",
        query="ESG 경영이란",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["ESG", "환경", "사회", "지배구조"],
        expected_answer="ESG는 환경(E), 사회(S), 지배구조(G)를 고려한 경영입니다.",
        relevant_sources=["ESG"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="ind_017",
        query="AI 반도체 시장 전망",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["AI", "반도체", "시장"],
        expected_answer=None,
        relevant_sources=["AI", "반도체"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="ind_018",
        query="탄소중립과 기업 영향",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["탄소중립", "기업"],
        expected_answer=None,
        relevant_sources=["탄소중립"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="ind_019",
        query="UAM 도심항공 모빌리티",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["UAM", "도심항공"],
        expected_answer=None,
        relevant_sources=["UAM"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="ind_020",
        query="로봇 산업 성장",
        category=QueryCategory.INDUSTRY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["로봇"],
        expected_answer=None,
        relevant_sources=["로봇"],
        difficulty="medium"
    ),

    # ========================================
    # 투자/재무 (15개)
    # ========================================
    EvaluationQuery(
        id="inv_001",
        query="삼성전자 배당금",
        category=QueryCategory.INVESTMENT,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["삼성전자", "배당"],
        expected_answer=None,
        relevant_sources=["삼성전자", "배당"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="inv_002",
        query="유상증자 절차",
        category=QueryCategory.INVESTMENT,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["유상증자"],
        expected_answer=None,
        relevant_sources=["유상증자"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="inv_003",
        query="자사주 매입 효과",
        category=QueryCategory.INVESTMENT,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["자사주", "매입"],
        expected_answer="자사주 매입은 주가 상승과 주주가치 제고 효과가 있습니다.",
        relevant_sources=["자사주"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="inv_004",
        query="액면분할이란",
        category=QueryCategory.INVESTMENT,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["액면분할"],
        expected_answer="액면분할은 주식의 액면가를 낮춰 주식 수를 늘리는 것입니다.",
        relevant_sources=["액면분할"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="inv_005",
        query="ROE 높은 기업",
        category=QueryCategory.INVESTMENT,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["ROE"],
        expected_answer=None,
        relevant_sources=["ROE"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="inv_006",
        query="EPS 성장률",
        category=QueryCategory.INVESTMENT,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["EPS", "성장"],
        expected_answer=None,
        relevant_sources=["EPS"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="inv_007",
        query="부채비율 낮은 기업",
        category=QueryCategory.INVESTMENT,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["부채비율"],
        expected_answer=None,
        relevant_sources=["부채비율"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="inv_008",
        query="현금흐름 우수 기업",
        category=QueryCategory.INVESTMENT,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["현금흐름"],
        expected_answer=None,
        relevant_sources=["현금흐름"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="inv_009",
        query="M&A 최근 사례",
        category=QueryCategory.INVESTMENT,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["M&A", "인수합병"],
        expected_answer=None,
        relevant_sources=["M&A"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="inv_010",
        query="스톡옵션 행사",
        category=QueryCategory.INVESTMENT,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["스톡옵션"],
        expected_answer=None,
        relevant_sources=["스톡옵션"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="inv_011",
        query="CB 전환사채란",
        category=QueryCategory.INVESTMENT,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["CB", "전환사채"],
        expected_answer="CB(전환사채)는 주식으로 전환할 수 있는 사채입니다.",
        relevant_sources=["전환사채"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="inv_012",
        query="BW 신주인수권부사채",
        category=QueryCategory.INVESTMENT,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["BW", "신주인수권"],
        expected_answer=None,
        relevant_sources=["BW"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="inv_013",
        query="우선주와 보통주 차이",
        category=QueryCategory.INVESTMENT,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["우선주", "보통주"],
        expected_answer="우선주는 배당 우선권이 있지만 의결권이 없습니다.",
        relevant_sources=["우선주"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="inv_014",
        query="무상증자 권리락",
        category=QueryCategory.INVESTMENT,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["무상증자", "권리락"],
        expected_answer=None,
        relevant_sources=["무상증자"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="inv_015",
        query="배당락일이란",
        category=QueryCategory.INVESTMENT,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["배당락"],
        expected_answer="배당락일은 배당 받을 권리가 사라지는 날입니다.",
        relevant_sources=["배당락"],
        difficulty="easy"
    ),

    # ========================================
    # 공시/규제 (10개)
    # ========================================
    EvaluationQuery(
        id="dis_001",
        query="분기보고서 제출 기한",
        category=QueryCategory.DISCLOSURE,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["분기보고서", "기한"],
        expected_answer="분기보고서는 분기 종료 후 45일 이내에 제출해야 합니다.",
        relevant_sources=["공시"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="dis_002",
        query="주요사항보고서 제출 사유",
        category=QueryCategory.DISCLOSURE,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["주요사항보고서"],
        expected_answer=None,
        relevant_sources=["주요사항보고서"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="dis_003",
        query="대량보유 보고 기준",
        category=QueryCategory.DISCLOSURE,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["대량보유", "5%"],
        expected_answer="지분 5% 이상 보유 시 대량보유 보고 의무가 있습니다.",
        relevant_sources=["대량보유"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="dis_004",
        query="내부자 거래 규제",
        category=QueryCategory.DISCLOSURE,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["내부자", "거래"],
        expected_answer=None,
        relevant_sources=["내부자거래"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="dis_005",
        query="공정공시 제도란",
        category=QueryCategory.DISCLOSURE,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["공정공시"],
        expected_answer="공정공시는 중요 정보를 모든 투자자에게 동시에 공개하는 제도입니다.",
        relevant_sources=["공정공시"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="dis_006",
        query="상장폐지 요건",
        category=QueryCategory.DISCLOSURE,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["상장폐지"],
        expected_answer=None,
        relevant_sources=["상장폐지"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="dis_007",
        query="감사의견 종류",
        category=QueryCategory.DISCLOSURE,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["감사의견", "적정", "한정"],
        expected_answer="감사의견에는 적정, 한정, 부적정, 의견거절이 있습니다.",
        relevant_sources=["감사의견"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="dis_008",
        query="투자주의 환기 종목",
        category=QueryCategory.DISCLOSURE,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["투자주의"],
        expected_answer=None,
        relevant_sources=["투자주의"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="dis_009",
        query="최대주주 변경 공시",
        category=QueryCategory.DISCLOSURE,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["최대주주", "변경"],
        expected_answer=None,
        relevant_sources=["최대주주"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="dis_010",
        query="조회공시 요청 사유",
        category=QueryCategory.DISCLOSURE,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["조회공시"],
        expected_answer="조회공시는 주가나 거래량 급변 시 거래소가 요청합니다.",
        relevant_sources=["조회공시"],
        difficulty="medium"
    ),

    # ========================================
    # 기업 정보 (10개)
    # ========================================
    EvaluationQuery(
        id="comp_001",
        query="삼성전자 주요 사업 부문",
        category=QueryCategory.COMPANY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["삼성전자", "반도체", "스마트폰", "디스플레이"],
        expected_answer="삼성전자 주요 사업은 반도체, 스마트폰, 디스플레이 등입니다.",
        relevant_sources=["삼성전자"],
        difficulty="easy"
    ),
    EvaluationQuery(
        id="comp_002",
        query="현대자동차그룹 계열사",
        category=QueryCategory.COMPANY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["현대자동차", "기아", "현대모비스"],
        expected_answer=None,
        relevant_sources=["현대자동차"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="comp_003",
        query="SK그룹 지배구조",
        category=QueryCategory.COMPANY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["SK", "지주회사"],
        expected_answer=None,
        relevant_sources=["SK"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="comp_004",
        query="카카오 자회사 현황",
        category=QueryCategory.COMPANY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["카카오", "자회사"],
        expected_answer=None,
        relevant_sources=["카카오"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="comp_005",
        query="네이버 해외 진출",
        category=QueryCategory.COMPANY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["네이버", "해외", "라인"],
        expected_answer=None,
        relevant_sources=["네이버"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="comp_006",
        query="LG그룹 지주회사",
        category=QueryCategory.COMPANY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["LG", "지주회사"],
        expected_answer=None,
        relevant_sources=["LG"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="comp_007",
        query="포스코 철강 생산량",
        category=QueryCategory.COMPANY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["POSCO", "철강"],
        expected_answer=None,
        relevant_sources=["POSCO"],
        difficulty="hard"
    ),
    EvaluationQuery(
        id="comp_008",
        query="셀트리온 주요 제품",
        category=QueryCategory.COMPANY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["셀트리온", "램시마"],
        expected_answer=None,
        relevant_sources=["셀트리온"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="comp_009",
        query="삼성바이오로직스 사업 모델",
        category=QueryCategory.COMPANY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["삼성바이오", "CMO", "위탁생산"],
        expected_answer=None,
        relevant_sources=["삼성바이오"],
        difficulty="medium"
    ),
    EvaluationQuery(
        id="comp_010",
        query="현대건설 주요 프로젝트",
        category=QueryCategory.COMPANY,
        answer_type=AnswerType.RELEVANT,
        expected_keywords=["현대건설"],
        expected_answer=None,
        relevant_sources=["현대건설"],
        difficulty="hard"
    ),
]


def get_evaluation_dataset() -> List[EvaluationQuery]:
    """평가 데이터셋 반환"""
    return EVALUATION_QUERIES


def get_queries_by_category(category: QueryCategory) -> List[EvaluationQuery]:
    """카테고리별 쿼리 반환"""
    return [q for q in EVALUATION_QUERIES if q.category == category]


def get_queries_by_difficulty(difficulty: str) -> List[EvaluationQuery]:
    """난이도별 쿼리 반환"""
    return [q for q in EVALUATION_QUERIES if q.difficulty == difficulty]


def save_dataset_to_json(output_path: Path) -> Path:
    """데이터셋을 JSON으로 저장"""
    data = {
        "version": "1.0",
        "total_queries": len(EVALUATION_QUERIES),
        "categories": {
            cat.value: len(get_queries_by_category(cat))
            for cat in QueryCategory
        },
        "difficulties": {
            diff: len(get_queries_by_difficulty(diff))
            for diff in ["easy", "medium", "hard"]
        },
        "queries": [asdict(q) for q in EVALUATION_QUERIES]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path


def print_dataset_stats():
    """데이터셋 통계 출력"""
    print("=" * 50)
    print("금융 RAG 평가 데이터셋 통계")
    print("=" * 50)
    print(f"\n총 쿼리 수: {len(EVALUATION_QUERIES)}")

    print("\n카테고리별:")
    for cat in QueryCategory:
        count = len(get_queries_by_category(cat))
        print(f"  - {cat.value}: {count}개")

    print("\n난이도별:")
    for diff in ["easy", "medium", "hard"]:
        count = len(get_queries_by_difficulty(diff))
        print(f"  - {diff}: {count}개")

    print("\n정답 유형별:")
    for ans_type in AnswerType:
        count = len([q for q in EVALUATION_QUERIES if q.answer_type == ans_type])
        print(f"  - {ans_type.value}: {count}개")


if __name__ == "__main__":
    print_dataset_stats()

    # JSON으로 저장
    output = Path(__file__).parent.parent.parent / "data" / "evaluation_dataset.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_dataset_to_json(output)
    print(f"\n데이터셋 저장: {output}")
