"""
=============================================================================
01. Google Gemini API 기초 - 백엔드 개발자를 위한 가이드
=============================================================================

[학습 목표]
- LLM API 호출 구조 이해 (REST API와 동일)
- Google Gemini API 사용법 습득
- 프롬프트 구성의 기본 원리

[백엔드 개발자에게 익숙한 비유]
- Gemini API = 외부 REST API 서비스
- API Key = 인증 토큰
- generate_content() = POST /generate 엔드포인트
- response = JSON Response Body

실행 방법:
    python notebooks/01_gemini_api_basics.py
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

# .env 파일에서 환경변수 로드
load_dotenv()

# API 키 설정
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def example_1_basic_call():
    """
    예제 1: 가장 기본적인 API 호출

    Spring에서 RestTemplate으로 외부 API 호출하는 것과 동일한 개념
    """
    print("=" * 60)
    print("예제 1: 기본 API 호출")
    print("=" * 60)

    # 모델 초기화 (HttpClient 생성과 유사)
    model = genai.GenerativeModel('gemini-2.0-flash')

    # API 호출 (POST 요청과 동일)
    response = model.generate_content("안녕하세요, 간단히 자기소개 해주세요.")

    # 응답 출력
    print(f"\n응답: {response.text}")

    # 메타데이터 확인
    if hasattr(response, 'usage_metadata'):
        print(f"\n[토큰 사용량]")
        print(f"  - 입력: {response.usage_metadata.prompt_token_count} tokens")
        print(f"  - 출력: {response.usage_metadata.candidates_token_count} tokens")


def example_2_system_instruction():
    """
    예제 2: System Instruction 사용

    System Instruction = AI의 역할/성격을 정의
    마치 서버 설정 파일처럼 AI의 동작 방식을 설정
    """
    print("\n" + "=" * 60)
    print("예제 2: System Instruction으로 역할 부여")
    print("=" * 60)

    # 모델 초기화 시 system_instruction 설정
    model = genai.GenerativeModel(
        'gemini-2.0-flash',
        system_instruction="""당신은 금융 전문가입니다.
다음 규칙을 따르세요:
1. 전문적이지만 이해하기 쉽게 설명
2. 필요시 예시를 들어 설명
3. 불확실한 정보는 명시적으로 언급"""
    )

    response = model.generate_content("ETF가 뭔가요?")
    print(f"\n응답:\n{response.text}")


def example_3_conversation():
    """
    예제 3: 대화 맥락 유지 (Chat)

    LLM은 기본적으로 Stateless (HTTP처럼)
    Gemini는 chat 객체로 대화 기록을 자동 관리
    """
    print("\n" + "=" * 60)
    print("예제 3: 대화 맥락 유지 (멀티턴)")
    print("=" * 60)

    model = genai.GenerativeModel(
        'gemini-2.0-flash',
        system_instruction="당신은 친절한 금융 상담사입니다. 간결하게 답변하세요."
    )

    # Chat 세션 시작 (세션 객체 생성)
    chat = model.start_chat(history=[])

    # 첫 번째 질문
    response1 = chat.send_message("주식과 채권의 차이가 뭔가요?")
    print(f"[질문 1] 주식과 채권의 차이가 뭔가요?")
    print(f"[답변 1] {response1.text}\n")

    # 두 번째 질문 (이전 맥락 자동 유지!)
    response2 = chat.send_message("그럼 초보자에게는 뭐가 더 좋아요?")
    print(f"[질문 2] 그럼 초보자에게는 뭐가 더 좋아요?")
    print(f"[답변 2] {response2.text}")

    # 대화 기록 확인
    print(f"\n[대화 기록 수]: {len(chat.history)}개 메시지")


def example_4_parameters():
    """
    예제 4: 생성 파라미터 조절

    temperature = 응답의 창의성/일관성 조절
    max_output_tokens = 응답 길이 제한
    """
    print("\n" + "=" * 60)
    print("예제 4: 파라미터 조절")
    print("=" * 60)

    prompt = "금융 투자의 핵심 원칙 3가지를 알려주세요."

    # 낮은 temperature = 일관되고 예측 가능한 응답
    print("\n[temperature=0.2] 보수적, 일관된 응답:")
    model_conservative = genai.GenerativeModel(
        'gemini-2.0-flash',
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=300
        )
    )
    response_low = model_conservative.generate_content(prompt)
    print(response_low.text)

    # 높은 temperature = 창의적, 다양한 응답
    print("\n[temperature=1.0] 창의적, 다양한 응답:")
    model_creative = genai.GenerativeModel(
        'gemini-2.0-flash',
        generation_config=genai.GenerationConfig(
            temperature=1.0,
            max_output_tokens=300
        )
    )
    response_high = model_creative.generate_content(prompt)
    print(response_high.text)


def example_5_structured_output():
    """
    예제 5: 구조화된 응답 받기 (JSON)

    실제 서비스에서는 자연어보다 JSON 응답이 유용
    파싱이 쉽고, 프론트엔드/다른 서비스와 연동 가능
    """
    print("\n" + "=" * 60)
    print("예제 5: JSON 형식 응답")
    print("=" * 60)

    model = genai.GenerativeModel(
        'gemini-2.0-flash',
        system_instruction="""당신은 금융 데이터 분석가입니다.
응답은 반드시 유효한 JSON 형식으로만 제공하세요.
다른 텍스트 없이 JSON만 출력하세요.
형식:
{
    "term": "용어",
    "definition": "정의 (한 문장)",
    "example": "실제 예시",
    "risk_level": "low 또는 medium 또는 high"
}"""
    )

    response = model.generate_content("P2P 대출에 대해 설명해주세요.")

    import json
    raw_response = response.text.strip()

    # JSON 블록 추출 (```json ... ``` 제거)
    if raw_response.startswith("```"):
        lines = raw_response.split("\n")
        raw_response = "\n".join(lines[1:-1])

    try:
        result = json.loads(raw_response)
        print(f"\n용어: {result.get('term')}")
        print(f"정의: {result.get('definition')}")
        print(f"예시: {result.get('example')}")
        print(f"위험도: {result.get('risk_level')}")
    except json.JSONDecodeError as e:
        print(f"\nJSON 파싱 실패: {e}")
        print(f"원본 응답:\n{response.text}")


def example_6_comparison_with_backend():
    """
    예제 6: 백엔드 개발 관점에서 LLM API 이해

    익숙한 백엔드 패턴과 LLM API 비교
    """
    print("\n" + "=" * 60)
    print("예제 6: 백엔드 개발자 관점 정리")
    print("=" * 60)

    comparison = """
    [백엔드 개발 vs LLM API 비교]

    ┌─────────────────┬──────────────────────┬────────────────────────┐
    │ 개념            │ 백엔드 (Spring/Node) │ LLM API                │
    ├─────────────────┼──────────────────────┼────────────────────────┤
    │ 클라이언트      │ RestTemplate/Axios   │ genai.GenerativeModel  │
    │ 인증            │ Bearer Token         │ API Key                │
    │ 요청            │ HTTP POST Body       │ generate_content()     │
    │ 응답            │ JSON Response        │ response.text          │
    │ 세션 관리       │ Session/Redis        │ chat.history           │
    │ 설정            │ application.yml      │ system_instruction     │
    │ 타임아웃        │ timeout 설정         │ timeout 파라미터       │
    │ 에러 핸들링     │ try-catch            │ try-catch              │
    └─────────────────┴──────────────────────┴────────────────────────┘

    [핵심 인사이트]
    - LLM API는 "텍스트 입력 → 텍스트 출력" REST API
    - 복잡해 보이지만 결국 HTTP 통신
    - 기존 백엔드 지식 그대로 활용 가능
    """
    print(comparison)

    # 실제 코드로 증명
    print("\n[실제 API 호출 - 백엔드 스타일로 작성]")

    # 1. 클라이언트 설정 (Bean 등록처럼)
    llm_client = genai.GenerativeModel('gemini-2.0-flash')

    # 2. 요청 데이터 구성 (DTO처럼)
    user_query = "한 줄로 머신러닝을 설명해주세요."

    # 3. API 호출 (Service 레이어처럼)
    try:
        response = llm_client.generate_content(user_query)
        result = response.text
        print(f"  요청: {user_query}")
        print(f"  응답: {result}")
    except Exception as e:
        print(f"  에러: {str(e)}")


if __name__ == "__main__":
    # API 키 확인
    if not os.getenv("GOOGLE_API_KEY"):
        print("=" * 60)
        print("오류: GOOGLE_API_KEY가 설정되지 않았습니다!")
        print("=" * 60)
        exit(1)

    print("\n" + "=" * 60)
    print("  Gemini API 기초 실습 시작")
    print("  백엔드 개발자를 위한 가이드")
    print("=" * 60)

    # 예제 실행
    example_1_basic_call()
    example_2_system_instruction()
    example_3_conversation()
    example_4_parameters()
    example_5_structured_output()
    example_6_comparison_with_backend()

    print("\n" + "=" * 60)
    print("  실습 완료!")
    print("=" * 60)
    print("\n[핵심 정리]")
    print("  1. LLM API = REST API (익숙한 패턴 그대로)")
    print("  2. System Instruction = AI의 역할/규칙 정의")
    print("  3. Chat = 대화 맥락 자동 관리")
    print("  4. temperature = 창의성 조절 (낮을수록 일관)")
    print("  5. JSON 응답 = 실제 서비스 연동의 핵심")
    print("\n[다음 단계]")
    print("  - 02_prompt_engineering.py: 프롬프트 기법 심화")
    print("  - 03_rag_basics.py: RAG 시스템 기초")
