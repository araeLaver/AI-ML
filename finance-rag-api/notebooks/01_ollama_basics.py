# -*- coding: utf-8 -*-
"""
01. Ollama (로컬 LLM) 기초 - 백엔드 개발자를 위한 가이드

실행: python notebooks/01_ollama_basics.py
"""

import ollama


def example_1_basic_call():
    """예제 1: 기본 API 호출"""
    print("=" * 60)
    print("예제 1: 기본 API 호출")
    print("=" * 60)

    response = ollama.chat(
        model='llama3.2',
        messages=[
            {'role': 'user', 'content': 'Hello, please introduce yourself briefly.'}
        ]
    )

    print(f"\n응답: {response.message.content}")
    print(f"\n[처리 시간]: {response.total_duration / 1e9:.2f}초")


def example_2_system_prompt():
    """예제 2: System Prompt로 역할 부여"""
    print("\n" + "=" * 60)
    print("예제 2: System Prompt로 역할 부여")
    print("=" * 60)

    response = ollama.chat(
        model='llama3.2',
        messages=[
            {
                'role': 'system',
                'content': 'You are a financial expert. Explain concepts simply in Korean.'
            },
            {
                'role': 'user',
                'content': 'What is ETF?'
            }
        ]
    )

    print(f"\n응답:\n{response.message.content}")


def example_3_conversation():
    """예제 3: 대화 맥락 유지 (멀티턴)"""
    print("\n" + "=" * 60)
    print("예제 3: 대화 맥락 유지")
    print("=" * 60)

    conversation = [
        {'role': 'system', 'content': 'You are a helpful assistant. Answer in Korean briefly.'}
    ]

    # 첫 번째 질문
    conversation.append({'role': 'user', 'content': 'What is the difference between stocks and bonds?'})
    response1 = ollama.chat(model='llama3.2', messages=conversation)
    conversation.append({'role': 'assistant', 'content': response1.message.content})

    print(f"[질문 1] 주식과 채권의 차이는?")
    print(f"[답변 1] {response1.message.content}\n")

    # 두 번째 질문 (맥락 유지)
    conversation.append({'role': 'user', 'content': 'Which is better for beginners?'})
    response2 = ollama.chat(model='llama3.2', messages=conversation)

    print(f"[질문 2] 초보자에게 뭐가 더 좋아요?")
    print(f"[답변 2] {response2.message.content}")


def example_4_parameters():
    """예제 4: 파라미터 조절"""
    print("\n" + "=" * 60)
    print("예제 4: 파라미터 조절 (temperature)")
    print("=" * 60)

    prompt = "List 3 key principles of investing."

    print("\n[temperature=0.2] 일관된 응답:")
    r1 = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.2, 'num_predict': 150}
    )
    print(r1.message.content)

    print("\n[temperature=1.0] 창의적 응답:")
    r2 = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 1.0, 'num_predict': 150}
    )
    print(r2.message.content)


def example_5_json_output():
    """예제 5: JSON 형식 응답"""
    print("\n" + "=" * 60)
    print("예제 5: JSON 형식 응답")
    print("=" * 60)

    response = ollama.chat(
        model='llama3.2',
        messages=[
            {
                'role': 'system',
                'content': 'Respond only in valid JSON format: {"term": "...", "definition": "...", "risk": "low/medium/high"}'
            },
            {
                'role': 'user',
                'content': 'Explain P2P lending'
            }
        ],
        options={'temperature': 0.1}
    )

    print(f"\n응답:\n{response.message.content}")

    import json
    try:
        raw = response.message.content.strip()
        if '```' in raw:
            raw = raw.split('```')[1].replace('json', '').strip()
        result = json.loads(raw)
        print(f"\n[파싱 성공]")
        print(f"  용어: {result.get('term')}")
        print(f"  정의: {result.get('definition')}")
        print(f"  위험: {result.get('risk')}")
    except:
        print("\n[JSON 파싱 실패 - 원본 출력됨]")


def example_6_streaming():
    """예제 6: 스트리밍 응답"""
    print("\n" + "=" * 60)
    print("예제 6: 스트리밍 응답")
    print("=" * 60)

    print("\n[실시간 출력]:")
    stream = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': 'Explain machine learning in 2 sentences.'}],
        stream=True
    )

    for chunk in stream:
        print(chunk.message.content, end='', flush=True)
    print()


def example_7_comparison():
    """예제 7: 클라우드 vs 로컬 비교"""
    print("\n" + "=" * 60)
    print("예제 7: 클라우드 API vs 로컬 LLM")
    print("=" * 60)

    print("""
    +---------------+--------------------+--------------------+
    | 항목          | 클라우드 API       | 로컬 LLM (Ollama)  |
    +---------------+--------------------+--------------------+
    | 비용          | 토큰당 과금        | 무료               |
    | 속도          | 매우 빠름          | PC 사양 의존       |
    | 품질          | 최고               | 좋음               |
    | 프라이버시    | 데이터 외부 전송   | 로컬 처리          |
    | 오프라인      | 불가               | 가능               |
    +---------------+--------------------+--------------------+

    [핵심] API 구조가 동일 -> 코드 재사용 가능!
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Ollama 로컬 LLM 기초 실습")
    print("=" * 60)

    # 서버 확인
    try:
        models = ollama.list()
        names = [m.model for m in models.models]
        print(f"\n[설치된 모델]: {names}")
    except Exception as e:
        print(f"\n[오류] Ollama 연결 실패: {e}")
        exit(1)

    # 실습 실행
    example_1_basic_call()
    example_2_system_prompt()
    example_3_conversation()
    example_4_parameters()
    example_5_json_output()
    example_6_streaming()
    example_7_comparison()

    print("\n" + "=" * 60)
    print("  실습 완료!")
    print("=" * 60)
    print("""
[핵심 정리]
  1. Ollama = localhost REST API (클라우드와 동일 구조)
  2. 무료 + 오프라인 + 프라이버시
  3. 개발/테스트에 최적
  4. 클라우드 API로 쉽게 전환 가능

[다음 단계]
  - 02_prompt_engineering.py
  - 03_rag_basics.py
""")
