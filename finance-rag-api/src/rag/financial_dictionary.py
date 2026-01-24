"""
금융 도메인 동의어 사전

한국 금융 시장에서 사용되는 용어, 약어, 영문 표현을 매핑합니다.
Query Expansion에서 검색 정확도 향상을 위해 사용됩니다.
"""

from typing import Dict, List, Set


# =============================================================================
# 재무 지표 (Financial Metrics)
# =============================================================================
FINANCIAL_METRICS = {
    # 수익성 지표
    "PER": ["주가수익비율", "P/E ratio", "Price to Earnings", "주가수익률", "피이알"],
    "PBR": ["주가순자산비율", "P/B ratio", "Price to Book", "피비알"],
    "PSR": ["주가매출비율", "P/S ratio", "Price to Sales", "피에스알"],
    "PCR": ["주가현금흐름비율", "P/C ratio", "Price to Cashflow"],
    "EV/EBITDA": ["이브이에비타", "기업가치배수"],

    "ROE": ["자기자본이익률", "Return on Equity", "자본수익률", "알오이"],
    "ROA": ["총자산이익률", "Return on Assets", "자산수익률", "알오에이"],
    "ROIC": ["투하자본수익률", "Return on Invested Capital"],
    "ROI": ["투자수익률", "Return on Investment", "알오아이"],

    "EPS": ["주당순이익", "Earnings Per Share", "이피에스"],
    "BPS": ["주당순자산", "Book value Per Share", "비피에스"],
    "DPS": ["주당배당금", "Dividend Per Share", "디피에스"],
    "SPS": ["주당매출액", "Sales Per Share"],

    # 수익 관련
    "EBITDA": ["감가상각전영업이익", "세전영업이익", "이비타", "에비타"],
    "EBIT": ["영업이익", "Operating Income", "이빗"],
    "EBT": ["세전이익", "Earnings Before Tax"],
    "NI": ["당기순이익", "Net Income", "순이익"],
    "GP": ["매출총이익", "Gross Profit", "총이익"],

    # 마진 지표
    "OPM": ["영업이익률", "Operating Profit Margin", "영업마진"],
    "NPM": ["순이익률", "Net Profit Margin", "순마진"],
    "GPM": ["매출총이익률", "Gross Profit Margin", "총이익률"],

    # 성장 지표
    "YoY": ["전년대비", "전년동기대비", "Year over Year", "연간성장률"],
    "QoQ": ["전분기대비", "Quarter over Quarter", "분기성장률"],
    "MoM": ["전월대비", "Month over Month", "월간성장률"],
    "CAGR": ["연평균성장률", "Compound Annual Growth Rate"],

    # 부채/재무건전성
    "DE": ["부채비율", "Debt to Equity", "디이비율"],
    "CR": ["유동비율", "Current Ratio"],
    "QR": ["당좌비율", "Quick Ratio"],
    "ICR": ["이자보상비율", "Interest Coverage Ratio"],
}

# =============================================================================
# 기업 약어 (Company Abbreviations)
# =============================================================================
COMPANY_ABBREVIATIONS = {
    # 대형주
    "삼전": ["삼성전자", "Samsung Electronics", "005930"],
    "하닉": ["SK하이닉스", "SK Hynix", "000660"],
    "엘지전": ["LG전자", "LG Electronics", "066570"],
    "현차": ["현대자동차", "현대차", "Hyundai Motor", "005380"],
    "기차": ["기아자동차", "기아", "Kia", "000270"],
    "셀트": ["삼성바이오로직스", "207940"],
    "네이버": ["NAVER", "035420"],
    "카카오": ["Kakao", "035720"],
    "포스코": ["POSCO홀딩스", "포스코홀딩스", "005490"],
    "KB": ["KB금융", "KB금융지주", "105560"],
    "신한": ["신한지주", "신한금융", "055550"],
    "하나": ["하나금융지주", "하나금융", "086790"],
    "우리": ["우리금융지주", "우리금융", "316140"],

    # 2차전지/배터리
    "엘지화": ["LG화학", "051910"],
    "엘지엔": ["LG에너지솔루션", "LGES", "373220"],
    "삼SDI": ["삼성SDI", "006400"],
    "에코프로": ["에코프로비엠", "에코프로", "247540"],
    "포엠": ["포스코퓨처엠", "003670"],

    # 바이오/제약
    "셀젠": ["셀트리온", "Celltrion", "068270"],
    "삼바": ["Samsung Biologics"],  # 셀트와 중복 방지
    "유한": ["유한양행", "000100"],
    "녹십자": ["녹십자", "006280"],

    # IT/플랫폼
    "쿠팡": ["Coupang", "CPNG"],
    "배민": ["배달의민족", "우아한형제들"],
    "토스": ["토스", "비바리퍼블리카", "Viva Republica"],
    "당근": ["당근마켓", "Daangn"],
    "야놀자": ["Yanolja"],
}

# =============================================================================
# 산업/섹터 용어 (Industry Terms)
# =============================================================================
INDUSTRY_TERMS = {
    # 반도체
    "HBM": ["고대역폭메모리", "High Bandwidth Memory", "에이치비엠"],
    "DRAM": ["디램", "Dynamic RAM"],
    "NAND": ["낸드", "낸드플래시", "NAND Flash"],
    "SSD": ["솔리드스테이트드라이브", "Solid State Drive"],
    "GPU": ["그래픽처리장치", "Graphics Processing Unit", "지피유"],
    "NPU": ["신경망처리장치", "Neural Processing Unit", "엔피유"],
    "AP": ["애플리케이션프로세서", "Application Processor"],
    "파운드리": ["Foundry", "반도체위탁생산"],
    "팹리스": ["Fabless", "반도체설계전문"],
    "EUV": ["극자외선", "Extreme Ultraviolet", "이유브이"],

    # AI/인공지능
    "AI반도체": ["인공지능반도체", "AI칩", "AI Chip", "AI Accelerator"],
    "LLM": ["대규모언어모델", "Large Language Model", "거대언어모델"],
    "GPT": ["지피티", "Generative Pre-trained Transformer"],
    "ML": ["머신러닝", "Machine Learning", "기계학습"],
    "DL": ["딥러닝", "Deep Learning", "심층학습"],

    # 2차전지/배터리
    "2차전지": ["이차전지", "배터리", "리튬이온배터리", "Secondary Battery"],
    "EV배터리": ["전기차배터리", "EV Battery"],
    "ESS": ["에너지저장장치", "Energy Storage System", "이에스에스"],
    "양극재": ["Cathode", "캐소드"],
    "음극재": ["Anode", "애노드"],
    "전해질": ["Electrolyte"],
    "분리막": ["Separator", "세퍼레이터"],
    "전고체": ["전고체배터리", "Solid State Battery", "SSB"],
    "LFP": ["리튬인산철", "Lithium Iron Phosphate"],
    "NCM": ["니켈코발트망간", "Nickel Cobalt Manganese"],
    "NCA": ["니켈코발트알루미늄"],

    # 전기차/모빌리티
    "EV": ["전기차", "Electric Vehicle", "이브이", "전기자동차"],
    "BEV": ["순수전기차", "Battery Electric Vehicle"],
    "HEV": ["하이브리드", "Hybrid Electric Vehicle", "하이브리드차"],
    "PHEV": ["플러그인하이브리드", "Plug-in Hybrid"],
    "FCEV": ["수소전기차", "Fuel Cell Electric Vehicle", "수소차"],
    "자율주행": ["Autonomous Driving", "Self-driving", "AD"],
    "ADAS": ["첨단운전자보조시스템", "Advanced Driver Assistance Systems"],

    # 디스플레이
    "OLED": ["올레드", "유기발광다이오드", "Organic LED"],
    "LCD": ["엘시디", "액정디스플레이", "Liquid Crystal Display"],
    "QD": ["퀀텀닷", "Quantum Dot", "양자점"],
    "마이크로LED": ["Micro LED", "μLED"],

    # 통신/네트워크
    "5G": ["5세대이동통신", "5th Generation"],
    "6G": ["6세대이동통신", "6th Generation"],
    "IoT": ["사물인터넷", "Internet of Things", "아이오티"],
    "클라우드": ["Cloud", "클라우드컴퓨팅", "Cloud Computing"],
    "데이터센터": ["Data Center", "DC", "IDC"],

    # 바이오/헬스케어
    "바이오시밀러": ["Biosimilar", "바이오의약품복제약"],
    "신약": ["New Drug", "신규의약품"],
    "임상": ["Clinical Trial", "임상시험"],
    "FDA": ["미국식품의약국", "Food and Drug Administration"],
    "EMA": ["유럽의약품청", "European Medicines Agency"],
    "mRNA": ["메신저RNA", "Messenger RNA"],
    "ADC": ["항체약물접합체", "Antibody-Drug Conjugate"],
    "CAR-T": ["카티세포치료제", "키메라항원수용체T세포"],

    # 에너지/친환경
    "RE100": ["재생에너지100", "Renewable Energy 100"],
    "탄소중립": ["Net Zero", "Carbon Neutral", "넷제로"],
    "ESG": ["이에스지", "환경사회지배구조", "Environmental Social Governance"],
    "태양광": ["Solar", "PV", "Photovoltaic"],
    "풍력": ["Wind Power", "Wind Energy"],
    "수소": ["Hydrogen", "수소에너지"],
    "그린수소": ["Green Hydrogen"],
}

# =============================================================================
# 공시/보고서 용어 (Disclosure Terms)
# =============================================================================
DISCLOSURE_TERMS = {
    "사업보고서": ["연간보고서", "Annual Report", "연차보고서"],
    "분기보고서": ["분기실적", "Quarterly Report", "분기실적보고서"],
    "반기보고서": ["반기실적", "Semi-annual Report", "반기실적보고서"],
    "감사보고서": ["Audit Report", "외부감사보고서"],
    "IR": ["투자자관계", "Investor Relations", "아이알"],
    "실적발표": ["Earnings Release", "실적공시"],
    "컨퍼런스콜": ["Conference Call", "실적발표회", "컨콜"],
    "가이던스": ["Guidance", "실적전망", "가이드라인"],
    "컨센서스": ["Consensus", "시장예상치", "애널리스트예상"],
    "어닝서프라이즈": ["Earnings Surprise", "실적서프라이즈"],
    "어닝쇼크": ["Earnings Shock", "실적쇼크"],
}

# =============================================================================
# 시장/거래 용어 (Market Terms)
# =============================================================================
MARKET_TERMS = {
    # 상장/공모
    "IPO": ["기업공개", "상장", "Initial Public Offering", "아이피오"],
    "유상증자": ["증자", "신주발행", "Rights Offering"],
    "무상증자": ["무상주", "주식배당", "Stock Dividend"],
    "CB": ["전환사채", "Convertible Bond", "씨비"],
    "BW": ["신주인수권부사채", "Bond with Warrant", "비더블유"],
    "DR": ["주식예탁증서", "Depositary Receipt", "디알"],
    "ADR": ["미국예탁증권", "American Depositary Receipt"],

    # 주가/시장
    "시총": ["시가총액", "Market Cap", "Market Capitalization"],
    "거래량": ["Trading Volume", "Volume", "볼륨"],
    "거래대금": ["Trading Value", "거래금액"],
    "52주신고가": ["52-week High", "신고가"],
    "52주신저가": ["52-week Low", "신저가"],
    "상한가": ["Upper Limit", "가격제한상한"],
    "하한가": ["Lower Limit", "가격제한하한"],
    "공매도": ["Short Selling", "숏셀링", "공매"],
    "대차거래": ["Securities Lending", "주식대차"],

    # 투자/펀드
    "ETF": ["상장지수펀드", "Exchange Traded Fund", "이티에프"],
    "ETN": ["상장지수증권", "Exchange Traded Note", "이티엔"],
    "인덱스펀드": ["Index Fund", "지수추종펀드"],
    "액티브펀드": ["Active Fund", "적극운용펀드"],
    "패시브펀드": ["Passive Fund", "소극운용펀드"],
    "헤지펀드": ["Hedge Fund"],
    "PE": ["사모펀드", "Private Equity", "피이"],
    "VC": ["벤처캐피탈", "Venture Capital", "브이씨"],

    # 거래소/지수
    "코스피": ["KOSPI", "Korea Composite Stock Price Index"],
    "코스닥": ["KOSDAQ", "Korea Securities Dealers Automated Quotations"],
    "나스닥": ["NASDAQ", "National Association of Securities Dealers Automated Quotations"],
    "다우": ["다우존스", "Dow Jones", "DJIA"],
    "S&P500": ["S&P 500", "에스앤피500", "스탠더드앤드푸어스500"],
}

# =============================================================================
# 기간 표현 (Time Expressions)
# =============================================================================
TIME_EXPRESSIONS = {
    "1Q": ["1분기", "Q1", "1/4분기", "제1분기"],
    "2Q": ["2분기", "Q2", "2/4분기", "제2분기"],
    "3Q": ["3분기", "Q3", "3/4분기", "제3분기"],
    "4Q": ["4분기", "Q4", "4/4분기", "제4분기"],
    "상반기": ["1H", "H1", "상반기실적"],
    "하반기": ["2H", "H2", "하반기실적"],
    "FY": ["회계연도", "Fiscal Year", "사업연도"],
    "YTD": ["연초이후", "Year to Date", "연초대비"],
    "TTM": ["최근12개월", "Trailing Twelve Months"],
}


# =============================================================================
# 통합 사전 생성 (Combined Dictionary)
# =============================================================================
def _build_combined_dictionary() -> Dict[str, List[str]]:
    """모든 카테고리를 통합한 사전 생성"""
    combined = {}

    for category in [
        FINANCIAL_METRICS,
        COMPANY_ABBREVIATIONS,
        INDUSTRY_TERMS,
        DISCLOSURE_TERMS,
        MARKET_TERMS,
        TIME_EXPRESSIONS,
    ]:
        combined.update(category)

    return combined


def _build_reverse_dictionary(forward_dict: Dict[str, List[str]]) -> Dict[str, str]:
    """역방향 매핑 사전 생성 (동의어 -> 표준 용어)"""
    reverse = {}

    for canonical, synonyms in forward_dict.items():
        # 표준 용어 자체도 포함
        reverse[canonical.lower()] = canonical

        for syn in synonyms:
            reverse[syn.lower()] = canonical

    return reverse


# 전역 사전 인스턴스
FINANCIAL_SYNONYMS: Dict[str, List[str]] = _build_combined_dictionary()
REVERSE_SYNONYMS: Dict[str, str] = _build_reverse_dictionary(FINANCIAL_SYNONYMS)


# =============================================================================
# 유틸리티 함수
# =============================================================================
def get_synonyms(term: str) -> Set[str]:
    """특정 용어의 모든 동의어 반환"""
    result = {term}
    term_upper = term.upper()
    term_lower = term.lower()

    # 정방향 검색
    if term_upper in FINANCIAL_SYNONYMS:
        result.update(FINANCIAL_SYNONYMS[term_upper])

    # 역방향 검색
    if term_lower in REVERSE_SYNONYMS:
        canonical = REVERSE_SYNONYMS[term_lower]
        result.add(canonical)
        if canonical in FINANCIAL_SYNONYMS:
            result.update(FINANCIAL_SYNONYMS[canonical])

    return result


def get_canonical_term(term: str) -> str:
    """동의어에서 표준 용어 반환"""
    term_lower = term.lower()
    return REVERSE_SYNONYMS.get(term_lower, term)


def get_all_terms() -> Set[str]:
    """사전에 등록된 모든 용어 반환"""
    all_terms = set()

    for canonical, synonyms in FINANCIAL_SYNONYMS.items():
        all_terms.add(canonical)
        all_terms.update(synonyms)

    return all_terms


def get_category_terms(category: str) -> Dict[str, List[str]]:
    """특정 카테고리의 용어만 반환"""
    categories = {
        "metrics": FINANCIAL_METRICS,
        "companies": COMPANY_ABBREVIATIONS,
        "industry": INDUSTRY_TERMS,
        "disclosure": DISCLOSURE_TERMS,
        "market": MARKET_TERMS,
        "time": TIME_EXPRESSIONS,
    }
    return categories.get(category, {})


def get_statistics() -> Dict[str, int]:
    """사전 통계 반환"""
    return {
        "total_canonical_terms": len(FINANCIAL_SYNONYMS),
        "total_synonyms": sum(len(syns) for syns in FINANCIAL_SYNONYMS.values()),
        "total_entries": len(REVERSE_SYNONYMS),
        "categories": {
            "metrics": len(FINANCIAL_METRICS),
            "companies": len(COMPANY_ABBREVIATIONS),
            "industry": len(INDUSTRY_TERMS),
            "disclosure": len(DISCLOSURE_TERMS),
            "market": len(MARKET_TERMS),
            "time": len(TIME_EXPRESSIONS),
        }
    }


# =============================================================================
# 모듈 테스트
# =============================================================================
if __name__ == "__main__":
    stats = get_statistics()
    print("=" * 50)
    print("금융 동의어 사전 통계")
    print("=" * 50)
    print(f"표준 용어 수: {stats['total_canonical_terms']}")
    print(f"총 동의어 수: {stats['total_synonyms']}")
    print(f"총 엔트리 수: {stats['total_entries']}")
    print()
    print("카테고리별:")
    for cat, count in stats['categories'].items():
        print(f"  - {cat}: {count}")
    print()
    print("예시:")
    print(f"  PER 동의어: {get_synonyms('PER')}")
    print(f"  삼전 동의어: {get_synonyms('삼전')}")
    print(f"  HBM 동의어: {get_synonyms('HBM')}")
    print(f"  주가수익비율 → {get_canonical_term('주가수익비율')}")
