"""
=============================================================================
01. LLM API 기초 - 백엔드 개발자를 위한 가이드
=============================================================================

[학습 목표]
- LLM API 호출이 결국 REST API 호출임을 이해
- OpenAI API의 구조 파악 (Spring의 RestTemplate과 유사)
- 기본적인 프롬프트 구성 방법

[백엔드 개발자에게 익숙한 비유]
- OpenAI API = 외부 REST API 서비스
- API Key = 인증 토큰
- messages = Request Body
- response = Response Body

실행 방법:
    python notebooks/01_llm_api_basics.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# .env 파일에서 환경변수 로드 (Spring의 application.properties와 유사)
load_dotenv()


def example_1_basic_call():
    """
    예제 1: 가장 기본적인 API 호출

    Spring에서 RestTemplate으로 외부 API 호출하는 것과 동일한 개념
    """
    print("=" * 60)
    print("예제 1: 기본 API 호출")
    print("=" * 60)

    # OpenAI 클라이언트 초기화 (HttpClient 생성과 유사)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # API 호출 (POST 요청과 동일)
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 사용할 모델 (저렴하고 빠름)
        messages=[
            {"role": "user", "content": "안녕하세요, 간단히 자기소개 해주세요."}
        ]
    )

    # 응답 파싱
    answer = response.choices[0].message.content
    print(f"\n응답: {answer}")

    # 토큰 사용량 확인 (비용 관리에 중요!)
    print(f"\n[토큰 사용량]")
    print(f"  - 입력: {response.usage.prompt_tokens} tokens")
    print(f"  - 출력: {response.usage.completion_tokens} tokens")
    print(f"  - 총합: {response.usage.total_tokens} tokens")


def example_2_system_prompt():
    """
    예제 2: System Prompt 사용

    System Prompt = AI의 역할/성격을 정의
    마치 서버 설정 파일처럼 AI의 동작 방식을 설정
    """
    print("\n" + "=" * 60)
    print("예제 2: System Prompt로 역할 부여")
    print("=" * 60)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # messages 배열의 구조:
    # - system: AI의 역할/규칙 정의 (한 번 설정)
    # - user: 사용자 입력
    # - assistant: AI 응답 (대화 기록용)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """당신은 금융 전문가입니다.
                다음 규칙을 따르세요:
                1. 전문적이지만 이해하기 쉽게 설명
                2. 필요시 예시를 들어 설명
                3. 불확실한 정보는 명시적으로 언급"""
            },
            {
                "role": "user",
                "content": "ETF가 뭔가요?"
            }
        ]
    )

    print(f"\n응답:\n{response.choices[0].message.content}")


def example_3_conversation():
    """
    예제 3: 대화 맥락 유지

    LLM은 기본적으로 Stateless (HTTP처럼)
    대화 기록을 직접 관리해야 함 (세션 관리와 유사)
    """
    print("\n" + "=" * 60)
    print("예제 3: 대화 맥락 유지 (멀티턴)")
    print("=" * 60)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 대화 기록을 배열로 관리 (세션 저장소 개념)
    conversation_history = [
        {
            "role": "system",
            "content": "당신은 친절한 금융 상담사입니다. 간결하게 답변하세요."
        }
    ]

    # 첫 번째 질문
    conversation_history.append({
        "role": "user",
        "content": "주식과 채권의 차이가 뭔가요?"
    })

    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history
    )

    assistant_reply1 = response1.choices[0].message.content
    conversation_history.append({
        "role": "assistant",
        "content": assistant_reply1
    })

    print(f"[질문 1] 주식과 채권의 차이가 뭔가요?")
    print(f"[답변 1] {assistant_reply1}\n")

    # 두 번째 질문 (이전 맥락 참조)
    conversation_history.append({
        "role": "user",
        "content": "그럼 초보자에게는 뭐가 더 좋아요?"
    })

    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversation_history
    )

    assistant_reply2 = response2.choices[0].message.content
    print(f"[질문 2] 그럼 초보자에게는 뭐가 더 좋아요?")
    print(f"[답변 2] {assistant_reply2}")

    # "그럼"이라는 맥락을 이해함 = 대화 기록이 전달되었기 때문


def example_4_parameters():
    """
    예제 4: 주요 파라미터 이해

    temperature, max_tokens 등 = 응답 품질/특성 조절
    마치 DB 쿼리의 LIMIT, ORDER BY 같은 조절 옵션
    """
    print("\n" + "=" * 60)
    print("예제 4: 파라미터 조절")
    print("=" * 60)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = "금융 투자의 핵심 원칙 3가지를 알려주세요."

    # 낮은 temperature = 일관되고 예측 가능한 응답
    print("\n[temperature=0.2] 보수적, 일관된 응답:")
    response_low = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # 0~2, 낮을수록 일관됨
        max_tokens=200    # 최대 응답 길이 제한
    )
    print(response_low.choices[0].message.content)

    # 높은 temperature = 창의적, 다양한 응답
    print("\n[temperature=1.0] 창의적, 다양한 응답:")
    response_high = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=200
    )
    print(response_high.choices[0].message.content)


def example_5_structured_output():
    """
    예제 5: 구조화된 응답 받기 (JSON)

    실제 서비스에서는 자연어보다 JSON 응답이 유용
    파싱이 쉽고, 프론트엔드/다른 서비스와 연동 가능
    """
    print("\n" + "=" * 60)
    print("예제 5: JSON 형식 응답")
    print("=" * 60)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """당신은 금융 데이터 분석가입니다.
                응답은 반드시 다음 JSON 형식으로만 제공하세요:
                {
                    "term": "용어",
                    "definition": "정의",
                    "example": "실제 예시",
                    "risk_level": "low/medium/high"
                }"""
            },
            {
                "role": "user",
                "content": "P2P 대출에 대해 설명해주세요."
            }
        ],
        response_format={"type": "json_object"}  # JSON 응답 강제
    )

    import json
    result = json.loads(response.choices[0].message.content)

    print(f"\n용어: {result.get('term')}")
    print(f"정의: {result.get('definition')}")
    print(f"예시: {result.get('example')}")
    print(f"위험도: {result.get('risk_level')}")


if __name__ == "__main__":
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("=" * 60)
        print("오류: OPENAI_API_KEY가 설정되지 않았습니다!")
        print("=" * 60)
        print("\n1. .env.example 파일을 .env로 복사하세요")
        print("2. .env 파일에 실제 API 키를 입력하세요")
        print("3. API 키는 https://platform.openai.com/api-keys 에서 발급")
        exit(1)

    print("\n" + "=" * 60)
    print("  LLM API 기초 실습 시작")
    print("  백엔드 개발자를 위한 가이드")
    print("=" * 60)

    # 예제 실행
    example_1_basic_call()
    example_2_system_prompt()
    example_3_conversation()
    example_4_parameters()
    example_5_structured_output()

    print("\n" + "=" * 60)
    print("  실습 완료!")
    print("=" * 60)
    print("\n[다음 단계]")
    print("  - 02_prompt_engineering.py: 프롬프트 기법 심화")
    print("  - 03_rag_basics.py: RAG 시스템 기초")
