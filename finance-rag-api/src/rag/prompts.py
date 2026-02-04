# -*- coding: utf-8 -*-
"""
프롬프트 템플릿 관리 모듈 (Phase 16)

중앙화된 프롬프트 관리, 버전 관리, 템플릿 조합 지원
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from datetime import datetime
import re


class PromptCategory(Enum):
    """프롬프트 카테고리"""
    RAG = "rag"
    SUMMARIZATION = "summarization"
    ANALYSIS = "analysis"
    TRANSLATION = "translation"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    FEW_SHOT = "few_shot"
    EXTRACTION = "extraction"
    COMPARISON = "comparison"


@dataclass
class PromptTemplate:
    """프롬프트 템플릿"""
    name: str
    version: str
    category: PromptCategory
    system_prompt: str
    user_prompt_template: str
    description: str = ""
    variables: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        # 템플릿에서 변수 자동 추출
        if not self.variables:
            self.variables = self._extract_variables()

    def _extract_variables(self) -> List[str]:
        """템플릿에서 {variable} 형태의 변수 추출"""
        pattern = r'\{(\w+)\}'
        system_vars = set(re.findall(pattern, self.system_prompt))
        user_vars = set(re.findall(pattern, self.user_prompt_template))
        return list(system_vars | user_vars)

    def format(self, **kwargs) -> Dict[str, str]:
        """변수를 채워서 프롬프트 생성"""
        # 누락된 변수 확인
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing variables: {missing}")

        return {
            "system": self.system_prompt.format(**kwargs),
            "user": self.user_prompt_template.format(**kwargs)
        }

    def with_examples(self, examples: List[Dict[str, str]]) -> "PromptTemplate":
        """Few-shot 예시 추가"""
        return PromptTemplate(
            name=self.name,
            version=self.version,
            category=self.category,
            system_prompt=self.system_prompt,
            user_prompt_template=self.user_prompt_template,
            description=self.description,
            variables=self.variables,
            examples=examples,
            created_at=self.created_at
        )


class PromptRegistry:
    """프롬프트 레지스트리 - 중앙화된 템플릿 관리"""

    _templates: Dict[str, Dict[str, PromptTemplate]] = {}  # name -> version -> template
    _default_versions: Dict[str, str] = {}  # name -> default version

    @classmethod
    def register(cls, template: PromptTemplate, is_default: bool = True) -> None:
        """템플릿 등록"""
        if template.name not in cls._templates:
            cls._templates[template.name] = {}

        cls._templates[template.name][template.version] = template

        if is_default:
            cls._default_versions[template.name] = template.version

    @classmethod
    def get(cls, name: str, version: Optional[str] = None) -> Optional[PromptTemplate]:
        """템플릿 조회"""
        if name not in cls._templates:
            return None

        if version is None:
            version = cls._default_versions.get(name)

        return cls._templates[name].get(version)

    @classmethod
    def list(cls) -> List[Dict[str, Any]]:
        """등록된 템플릿 목록"""
        result = []
        for name, versions in cls._templates.items():
            default_version = cls._default_versions.get(name)
            for version, template in versions.items():
                result.append({
                    "name": name,
                    "version": version,
                    "category": template.category.value,
                    "is_default": version == default_version,
                    "description": template.description,
                    "variables": template.variables
                })
        return result

    @classmethod
    def get_by_category(cls, category: PromptCategory) -> List[PromptTemplate]:
        """카테고리별 템플릿 조회"""
        result = []
        for name, versions in cls._templates.items():
            default_version = cls._default_versions.get(name)
            if default_version and default_version in versions:
                template = versions[default_version]
                if template.category == category:
                    result.append(template)
        return result


# ============================================================
# 기본 프롬프트 템플릿 정의
# ============================================================

# RAG 기본 템플릿
RAG_SYSTEM_PROMPT = """당신은 금융 전문 상담 AI입니다.

역할:
- 제공된 문서만을 기반으로 정확하게 답변합니다
- 금융 용어를 쉽게 설명합니다
- 투자 조언이 아닌 정보 제공임을 명시합니다

규칙:
1. 문서에 없는 내용은 "해당 정보가 제공된 문서에 없습니다"라고 답하세요
2. 추측하거나 지어내지 마세요
3. 숫자나 수치는 문서 그대로 인용하세요
4. 답변 마지막에 참조 문서를 표시하세요

주의:
- 이 정보는 투자 권유가 아닙니다
- 실제 투자 결정은 전문가와 상담하세요"""

RAG_USER_TEMPLATE = """[참고 문서]
{context}

[사용자 질문]
{question}

[답변]"""

RAG_TEMPLATE = PromptTemplate(
    name="rag_default",
    version="1.0.0",
    category=PromptCategory.RAG,
    system_prompt=RAG_SYSTEM_PROMPT,
    user_prompt_template=RAG_USER_TEMPLATE,
    description="기본 RAG 프롬프트 - 금융 문서 기반 Q&A"
)


# Chain-of-Thought 템플릿
COT_SYSTEM_PROMPT = """당신은 금융 분석 전문가입니다.

복잡한 질문에 대해 단계별로 사고하여 답변합니다:
1. 먼저 질문을 분석합니다
2. 관련 정보를 정리합니다
3. 논리적으로 추론합니다
4. 최종 결론을 도출합니다

규칙:
- 각 단계를 명확히 구분하여 설명하세요
- 계산이 필요하면 과정을 보여주세요
- 문서에 없는 정보는 추측하지 마세요"""

COT_USER_TEMPLATE = """[참고 문서]
{context}

[질문]
{question}

[단계별 분석]
1단계 - 질문 분석:
"""

COT_TEMPLATE = PromptTemplate(
    name="chain_of_thought",
    version="1.0.0",
    category=PromptCategory.CHAIN_OF_THOUGHT,
    system_prompt=COT_SYSTEM_PROMPT,
    user_prompt_template=COT_USER_TEMPLATE,
    description="Chain-of-Thought 프롬프트 - 복잡한 분석 질문용"
)


# 요약 템플릿
SUMMARY_SYSTEM_PROMPT = """당신은 금융 문서 요약 전문가입니다.

요약 원칙:
- 핵심 정보만 추출합니다
- 숫자와 수치는 정확히 유지합니다
- 객관적인 톤을 유지합니다
- 원문의 의도를 왜곡하지 않습니다

출력 형식:
- 3-5개의 핵심 포인트로 요약
- 각 포인트는 한 문장으로 작성"""

SUMMARY_USER_TEMPLATE = """[문서]
{document}

[요약 요청]
{instruction}

[요약]"""

SUMMARY_TEMPLATE = PromptTemplate(
    name="summarization",
    version="1.0.0",
    category=PromptCategory.SUMMARIZATION,
    system_prompt=SUMMARY_SYSTEM_PROMPT,
    user_prompt_template=SUMMARY_USER_TEMPLATE,
    description="문서 요약 프롬프트"
)


# 비교 분석 템플릿
COMPARISON_SYSTEM_PROMPT = """당신은 금융 비교 분석 전문가입니다.

비교 분석 원칙:
- 동일한 기준으로 비교합니다
- 장단점을 균형있게 제시합니다
- 수치는 정확히 인용합니다
- 최종 판단은 사용자에게 맡깁니다

출력 형식:
| 항목 | {item1} | {item2} |
|------|---------|---------|
...

결론 및 고려사항을 별도로 작성합니다."""

COMPARISON_USER_TEMPLATE = """[비교 대상 문서들]
{documents}

[비교 항목]
{comparison_criteria}

[비교 분석]"""

COMPARISON_TEMPLATE = PromptTemplate(
    name="comparison",
    version="1.0.0",
    category=PromptCategory.COMPARISON,
    system_prompt=COMPARISON_SYSTEM_PROMPT,
    user_prompt_template=COMPARISON_USER_TEMPLATE,
    description="비교 분석 프롬프트"
)


# 정보 추출 템플릿
EXTRACTION_SYSTEM_PROMPT = """당신은 금융 정보 추출 전문가입니다.

추출 규칙:
- 요청된 정보만 정확히 추출합니다
- 없는 정보는 "N/A"로 표시합니다
- JSON 형식으로 출력합니다
- 추론이나 추측은 하지 않습니다"""

EXTRACTION_USER_TEMPLATE = """[문서]
{document}

[추출 항목]
{fields}

[추출 결과 (JSON)]
```json
"""

EXTRACTION_TEMPLATE = PromptTemplate(
    name="extraction",
    version="1.0.0",
    category=PromptCategory.EXTRACTION,
    system_prompt=EXTRACTION_SYSTEM_PROMPT,
    user_prompt_template=EXTRACTION_USER_TEMPLATE,
    description="구조화된 정보 추출 프롬프트"
)


# Few-shot 기본 템플릿
FEW_SHOT_SYSTEM_PROMPT = """당신은 금융 전문 상담 AI입니다.
아래 예시들을 참고하여 일관된 형식으로 답변하세요.

{examples}

위 예시들처럼 답변해주세요."""

FEW_SHOT_USER_TEMPLATE = """[참고 문서]
{context}

[질문]
{question}

[답변]"""

FEW_SHOT_TEMPLATE = PromptTemplate(
    name="few_shot",
    version="1.0.0",
    category=PromptCategory.FEW_SHOT,
    system_prompt=FEW_SHOT_SYSTEM_PROMPT,
    user_prompt_template=FEW_SHOT_USER_TEMPLATE,
    description="Few-shot 학습 프롬프트"
)


# 엄격한 RAG 템플릿 (환각 최소화)
STRICT_RAG_SYSTEM_PROMPT = """당신은 금융 문서 기반 답변 시스템입니다.

⚠️ 절대 규칙:
1. 제공된 문서에 명시적으로 있는 정보만 사용합니다
2. 문서에 없는 정보는 절대 생성하지 않습니다
3. 불확실한 경우 "문서에서 해당 정보를 찾을 수 없습니다"라고 답합니다
4. 모든 수치와 사실은 문서 원문 그대로 인용합니다

출력 형식:
- 답변: [문서 기반 답변]
- 출처: [인용한 문서명]
- 신뢰도: [높음/중간/낮음]

주의: 이 정보는 투자 권유가 아닙니다."""

STRICT_RAG_USER_TEMPLATE = """[제공된 문서]
{context}

[사용자 질문]
{question}

[문서 기반 답변]"""

STRICT_RAG_TEMPLATE = PromptTemplate(
    name="rag_strict",
    version="1.0.0",
    category=PromptCategory.RAG,
    system_prompt=STRICT_RAG_SYSTEM_PROMPT,
    user_prompt_template=STRICT_RAG_USER_TEMPLATE,
    description="엄격한 RAG 프롬프트 - 환각 최소화"
)


# 대화형 RAG 템플릿
CONVERSATIONAL_RAG_SYSTEM_PROMPT = """당신은 친절한 금융 상담사입니다.

역할:
- 사용자와 자연스러운 대화를 합니다
- 전문 용어는 쉽게 풀어서 설명합니다
- 필요시 추가 질문을 유도합니다
- 문서 기반으로만 답변합니다

대화 스타일:
- 친근하지만 전문적인 톤
- 이해하기 쉬운 비유 활용
- 핵심을 먼저, 상세 설명은 후에

규칙:
- 문서에 없는 내용은 솔직히 모른다고 합니다
- 투자 조언이 아닌 정보 제공임을 명시합니다"""

CONVERSATIONAL_RAG_USER_TEMPLATE = """[대화 기록]
{history}

[참고 문서]
{context}

[현재 질문]
{question}

[상담사 답변]"""

CONVERSATIONAL_RAG_TEMPLATE = PromptTemplate(
    name="rag_conversational",
    version="1.0.0",
    category=PromptCategory.RAG,
    system_prompt=CONVERSATIONAL_RAG_SYSTEM_PROMPT,
    user_prompt_template=CONVERSATIONAL_RAG_USER_TEMPLATE,
    description="대화형 RAG 프롬프트 - 멀티턴 대화용"
)


# ============================================================
# 템플릿 등록
# ============================================================

def register_default_templates():
    """기본 템플릿 등록"""
    PromptRegistry.register(RAG_TEMPLATE, is_default=True)
    PromptRegistry.register(COT_TEMPLATE, is_default=True)
    PromptRegistry.register(SUMMARY_TEMPLATE, is_default=True)
    PromptRegistry.register(COMPARISON_TEMPLATE, is_default=True)
    PromptRegistry.register(EXTRACTION_TEMPLATE, is_default=True)
    PromptRegistry.register(FEW_SHOT_TEMPLATE, is_default=True)
    PromptRegistry.register(STRICT_RAG_TEMPLATE, is_default=False)
    PromptRegistry.register(CONVERSATIONAL_RAG_TEMPLATE, is_default=False)


# 모듈 로드 시 기본 템플릿 등록
register_default_templates()


# ============================================================
# 편의 함수
# ============================================================

def get_rag_prompt(
    context: str,
    question: str,
    template_name: str = "rag_default",
    version: Optional[str] = None
) -> Dict[str, str]:
    """RAG 프롬프트 생성 헬퍼"""
    template = PromptRegistry.get(template_name, version)
    if not template:
        template = RAG_TEMPLATE

    return template.format(context=context, question=question)


def get_cot_prompt(context: str, question: str) -> Dict[str, str]:
    """Chain-of-Thought 프롬프트 생성 헬퍼"""
    return COT_TEMPLATE.format(context=context, question=question)


def get_summary_prompt(document: str, instruction: str = "핵심 내용을 요약해주세요") -> Dict[str, str]:
    """요약 프롬프트 생성 헬퍼"""
    return SUMMARY_TEMPLATE.format(document=document, instruction=instruction)


def create_few_shot_examples(examples: List[Dict[str, str]]) -> str:
    """Few-shot 예시 문자열 생성"""
    formatted = []
    for i, ex in enumerate(examples, 1):
        formatted.append(f"예시 {i}:")
        formatted.append(f"질문: {ex.get('question', '')}")
        formatted.append(f"답변: {ex.get('answer', '')}")
        formatted.append("")
    return "\n".join(formatted)
