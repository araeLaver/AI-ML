# -*- coding: utf-8 -*-
"""
02. 프롬프트 엔지니어링 - 실무에서 바로 쓰는 기법

[왜 중요한가?]
- 동일한 LLM이라도 프롬프트에 따라 품질이 10배 차이
- RAG 시스템의 핵심 = 좋은 프롬프트 설계
- 면접에서 "프롬프트 어떻게 설계했나요?" 질문 대비

[포트폴리오 연결]
- 금융 문서 RAG에서 사용할 프롬프트 패턴 학습
- 실제 서비스에서 쓰는 기법들

실행: python notebooks/02_prompt_engineering.py
"""

import ollama
import json


def technique_1_role_prompting():
    """
    기법 1: Role Prompting (역할 부여)

    [핵심] AI에게 전문가 역할을 부여하면 답변 품질 향상
    [포트폴리오 적용] 금융 전문가 역할 → 금융 용어 정확도 향상
    """
    print("=" * 60)
    print("기법 1: Role Prompting (역할 부여)")
    print("=" * 60)

    question = "What are the risks of investing in REITs?"

    # 역할 없이
    print("\n[역할 없이]")
    r1 = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': question}],
        options={'num_predict': 150}
    )
    print(r1.message.content)

    # 역할 부여
    print("\n[금융 전문가 역할 부여]")
    r2 = ollama.chat(
        model='llama3.2',
        messages=[
            {
                'role': 'system',
                'content': '''You are a senior financial advisor with 20 years of experience.
You specialize in real estate investments and risk assessment.
Always explain risks in practical terms with specific examples.
Answer in a structured format.'''
            },
            {'role': 'user', 'content': question}
        ],
        options={'num_predict': 200}
    )
    print(r2.message.content)

    print("\n[포인트] 역할 부여 시 더 전문적이고 구조화된 답변")


def technique_2_few_shot():
    """
    기법 2: Few-shot Prompting (예시 제공)

    [핵심] 원하는 출력 형식의 예시를 보여주면 일관성 향상
    [포트폴리오 적용] 금융 용어 설명 형식 통일
    """
    print("\n" + "=" * 60)
    print("기법 2: Few-shot Prompting (예시 제공)")
    print("=" * 60)

    # Zero-shot (예시 없이)
    print("\n[Zero-shot] 예시 없이:")
    r1 = ollama.chat(
        model='llama3.2',
        messages=[
            {'role': 'user', 'content': 'Explain what a mutual fund is.'}
        ],
        options={'num_predict': 100}
    )
    print(r1.message.content)

    # Few-shot (예시 제공)
    print("\n[Few-shot] 예시 제공:")
    r2 = ollama.chat(
        model='llama3.2',
        messages=[
            {
                'role': 'system',
                'content': '''Explain financial terms in this exact format:

Term: ETF
Definition: A basket of securities traded on stock exchanges like individual stocks.
Key Feature: Low fees, diversification, real-time trading.
Risk Level: Medium
Best For: Beginners seeking diversified exposure.

Term: Bond
Definition: A loan to a company or government that pays fixed interest.
Key Feature: Fixed income, principal protection, predictable returns.
Risk Level: Low
Best For: Conservative investors seeking stable income.

Now explain the following term in the same format:'''
            },
            {'role': 'user', 'content': 'Mutual Fund'}
        ],
        options={'num_predict': 150}
    )
    print(r2.message.content)

    print("\n[포인트] 예시를 보여주면 형식이 일관됨 → JSON 출력에 필수")


def technique_3_chain_of_thought():
    """
    기법 3: Chain-of-Thought (단계별 사고)

    [핵심] "단계별로 생각해봐"라고 하면 복잡한 문제 해결력 향상
    [포트폴리오 적용] 금융 계산, 복잡한 분석에 활용
    """
    print("\n" + "=" * 60)
    print("기법 3: Chain-of-Thought (단계별 사고)")
    print("=" * 60)

    problem = """
    An investor has $100,000. They want to allocate:
    - 60% to stocks (expected return 10%)
    - 30% to bonds (expected return 5%)
    - 10% to cash (expected return 2%)
    What is the expected portfolio return?
    """

    # 일반 요청
    print("\n[일반 요청]")
    r1 = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': problem + "\nAnswer:"}],
        options={'num_predict': 100}
    )
    print(r1.message.content)

    # Chain-of-Thought
    print("\n[Chain-of-Thought]")
    r2 = ollama.chat(
        model='llama3.2',
        messages=[
            {'role': 'user', 'content': problem + "\nLet's solve this step by step:"}
        ],
        options={'num_predict': 200}
    )
    print(r2.message.content)

    print("\n[포인트] 'step by step' 추가만으로 정확도 향상")


def technique_4_output_format():
    """
    기법 4: Output Format Control (출력 형식 지정)

    [핵심] JSON, 마크다운 등 원하는 형식 명시
    [포트폴리오 적용] RAG API 응답을 JSON으로 반환
    """
    print("\n" + "=" * 60)
    print("기법 4: Output Format Control")
    print("=" * 60)

    query = "Analyze Samsung Electronics as an investment."

    # JSON 형식 강제
    print("\n[JSON 형식 출력]")
    r1 = ollama.chat(
        model='llama3.2',
        messages=[
            {
                'role': 'system',
                'content': '''You are a stock analyst.
Output ONLY valid JSON, no other text.
Format:
{
    "company": "company name",
    "sector": "industry sector",
    "strengths": ["strength1", "strength2"],
    "risks": ["risk1", "risk2"],
    "recommendation": "buy/hold/sell",
    "confidence": "high/medium/low"
}'''
            },
            {'role': 'user', 'content': query}
        ],
        options={'temperature': 0.1, 'num_predict': 200}
    )
    print(r1.message.content)

    # 파싱 테스트
    try:
        raw = r1.message.content.strip()
        if '```' in raw:
            raw = raw.split('```')[1].replace('json', '').strip()
        data = json.loads(raw)
        print(f"\n[파싱 성공] 회사: {data.get('company')}, 추천: {data.get('recommendation')}")
    except:
        print("\n[파싱 실패]")


def technique_5_context_injection():
    """
    기법 5: Context Injection (문맥 주입)

    [핵심] 외부 정보를 프롬프트에 삽입 → RAG의 핵심!
    [포트폴리오 적용] 검색된 문서를 프롬프트에 주입
    """
    print("\n" + "=" * 60)
    print("기법 5: Context Injection (RAG 핵심)")
    print("=" * 60)

    # 가상의 검색된 문서 (실제 RAG에서는 벡터 DB에서 검색)
    retrieved_context = """
    [Document 1: Company Policy]
    Retirement Plan: Employees can contribute up to 10% of salary.
    Company matches 50% of contributions up to 6%.
    Vesting period: 3 years for full company match.

    [Document 2: 2024 Updates]
    New benefit: Health savings account (HSA) option added.
    Maximum HSA contribution: $4,150 for individuals.
    """

    user_question = "How much will the company match if I contribute 8% of my salary?"

    # 컨텍스트 없이 (환각 발생 가능)
    print("\n[컨텍스트 없이] - 환각 위험")
    r1 = ollama.chat(
        model='llama3.2',
        messages=[{'role': 'user', 'content': user_question}],
        options={'num_predict': 100}
    )
    print(r1.message.content)

    # 컨텍스트 주입 (RAG 방식)
    print("\n[컨텍스트 주입] - RAG 방식")
    r2 = ollama.chat(
        model='llama3.2',
        messages=[
            {
                'role': 'system',
                'content': f'''Answer based ONLY on the provided context.
If the answer is not in the context, say "I don't have that information."

Context:
{retrieved_context}'''
            },
            {'role': 'user', 'content': user_question}
        ],
        options={'num_predict': 150}
    )
    print(r2.message.content)

    print("\n[포인트] 이것이 RAG의 핵심 - 검색된 문서를 프롬프트에 주입")


def technique_6_rag_prompt_template():
    """
    기법 6: RAG 프롬프트 템플릿 (포트폴리오용)

    [핵심] 실제 금융 RAG 시스템에서 사용할 프롬프트
    [포트폴리오 적용] 그대로 복사해서 사용
    """
    print("\n" + "=" * 60)
    print("기법 6: RAG 프롬프트 템플릿 (실전용)")
    print("=" * 60)

    # 실제 RAG 시스템용 프롬프트 템플릿
    RAG_SYSTEM_PROMPT = """You are a financial document assistant for a Korean financial services company.

ROLE:
- Answer questions based ONLY on the provided documents
- Explain financial terms in simple Korean when needed
- Always cite the source document

RULES:
1. If the answer is not in the documents, say "해당 정보를 찾을 수 없습니다."
2. Do not make up information
3. If multiple documents are relevant, synthesize them
4. For numerical data, quote exactly from the document

OUTPUT FORMAT:
- Answer in Korean
- Include source citation: [출처: document name]
- Keep answers concise but complete

CONTEXT DOCUMENTS:
{context}

USER QUESTION: {question}"""

    # 테스트
    test_context = """
    [삼성전자 2024년 1분기 실적]
    매출: 71.9조원 (전년 동기 대비 +12.8%)
    영업이익: 6.6조원 (전년 동기 대비 +932%)
    반도체 부문: HBM 수요 증가로 메모리 실적 개선
    """

    test_question = "삼성전자 1분기 영업이익이 얼마야?"

    prompt = RAG_SYSTEM_PROMPT.format(context=test_context, question=test_question)

    print("\n[RAG 프롬프트 템플릿 테스트]")
    response = ollama.chat(
        model='llama3.2',
        messages=[
            {'role': 'system', 'content': prompt.split("USER QUESTION:")[0]},
            {'role': 'user', 'content': test_question}
        ],
        options={'num_predict': 150}
    )
    print(response.message.content)

    print("\n" + "-" * 60)
    print("[실전 RAG 프롬프트 템플릿]")
    print("-" * 60)
    print(RAG_SYSTEM_PROMPT[:500] + "...")


def summary():
    """학습 내용 요약"""
    print("\n" + "=" * 60)
    print("프롬프트 엔지니어링 핵심 정리")
    print("=" * 60)

    print("""
    +-------------------+--------------------------------+------------------+
    | 기법              | 설명                           | 포트폴리오 활용  |
    +-------------------+--------------------------------+------------------+
    | Role Prompting    | 전문가 역할 부여               | 금융 전문가 설정 |
    | Few-shot          | 예시로 형식 통일               | JSON 출력 형식   |
    | Chain-of-Thought  | 단계별 사고 유도               | 복잡한 분석      |
    | Output Format     | JSON/마크다운 형식 지정        | API 응답 구조화  |
    | Context Injection | 외부 문서 주입                 | RAG 핵심 기법    |
    | RAG Template      | 실전용 프롬프트 템플릿         | 그대로 사용      |
    +-------------------+--------------------------------+------------------+

    [면접 대비 핵심 질문]
    Q: "RAG에서 환각(Hallucination)을 어떻게 줄이나요?"
    A: "Context Injection으로 검색된 문서만 참조하도록 제한하고,
        '문서에 없으면 모른다고 답하라'는 규칙을 명시합니다."

    Q: "프롬프트 엔지니어링에서 가장 중요한 것은?"
    A: "명확한 역할 정의, 출력 형식 지정, 그리고 예시 제공입니다.
        특히 JSON 출력이 필요한 서비스에서는 Few-shot이 필수입니다."
    """)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  프롬프트 엔지니어링 실습")
    print("  실무에서 바로 쓰는 6가지 기법")
    print("=" * 60)

    technique_1_role_prompting()
    technique_2_few_shot()
    technique_3_chain_of_thought()
    technique_4_output_format()
    technique_5_context_injection()
    technique_6_rag_prompt_template()
    summary()

    print("\n[다음 단계] 03_rag_basics.py - RAG 시스템 구축")
