# -*- coding: utf-8 -*-
"""
도구 정의 및 레지스트리 모듈

[기능]
- 도구 정의 및 스키마
- 도구 레지스트리
- 도구 검증
"""

import inspect
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """파라미터 타입"""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """도구 파라미터"""
    name: str
    param_type: ParameterType
    description: str = ""
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None
    items_type: Optional[ParameterType] = None  # Array인 경우

    def to_schema(self) -> Dict[str, Any]:
        """JSON Schema로 변환"""
        schema = {
            "type": self.param_type.value,
            "description": self.description,
        }

        if self.enum:
            schema["enum"] = self.enum

        if self.param_type == ParameterType.ARRAY and self.items_type:
            schema["items"] = {"type": self.items_type.value}

        return schema


@dataclass
class ToolResult:
    """도구 실행 결과"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "metadata": self.metadata,
        }


class Tool(ABC):
    """
    도구 기본 클래스

    에이전트가 사용할 수 있는 도구 정의
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[List[ToolParameter]] = None,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters or []

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """도구 실행"""
        pass

    def validate_params(self, params: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """파라미터 검증"""
        for param in self.parameters:
            if param.required and param.name not in params:
                return False, f"Required parameter missing: {param.name}"

            if param.name in params:
                value = params[param.name]

                # 타입 검증
                if param.param_type == ParameterType.STRING and not isinstance(value, str):
                    return False, f"Parameter {param.name} must be string"
                elif param.param_type == ParameterType.INTEGER and not isinstance(value, int):
                    return False, f"Parameter {param.name} must be integer"
                elif param.param_type == ParameterType.NUMBER and not isinstance(value, (int, float)):
                    return False, f"Parameter {param.name} must be number"
                elif param.param_type == ParameterType.BOOLEAN and not isinstance(value, bool):
                    return False, f"Parameter {param.name} must be boolean"
                elif param.param_type == ParameterType.ARRAY and not isinstance(value, list):
                    return False, f"Parameter {param.name} must be array"

                # Enum 검증
                if param.enum and value not in param.enum:
                    return False, f"Parameter {param.name} must be one of {param.enum}"

        return True, None

    def get_schema(self) -> Dict[str, Any]:
        """OpenAI/Anthropic 호환 스키마"""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class FunctionTool(Tool):
    """
    함수 기반 도구

    Python 함수를 도구로 래핑
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[List[ToolParameter]] = None,
    ):
        self.func = func
        func_name = name or func.__name__
        func_desc = description or (func.__doc__ or "").strip().split("\n")[0]

        # 자동 파라미터 추출
        if parameters is None:
            parameters = self._extract_parameters(func)

        super().__init__(func_name, func_desc, parameters)

    def _extract_parameters(self, func: Callable) -> List[ToolParameter]:
        """함수 시그니처에서 파라미터 추출"""
        sig = inspect.signature(func)
        params = []

        type_mapping = {
            str: ParameterType.STRING,
            int: ParameterType.INTEGER,
            float: ParameterType.NUMBER,
            bool: ParameterType.BOOLEAN,
            list: ParameterType.ARRAY,
            dict: ParameterType.OBJECT,
        }

        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue

            # 타입 힌트에서 타입 추출
            param_type = ParameterType.STRING
            if param.annotation != inspect.Parameter.empty:
                param_type = type_mapping.get(param.annotation, ParameterType.STRING)

            # 기본값 확인
            has_default = param.default != inspect.Parameter.empty
            default = param.default if has_default else None

            params.append(ToolParameter(
                name=name,
                param_type=param_type,
                description=f"Parameter: {name}",
                required=not has_default,
                default=default,
            ))

        return params

    def execute(self, **kwargs) -> ToolResult:
        """함수 실행"""
        start_time = time.time()

        try:
            # 기본값 적용
            for param in self.parameters:
                if param.name not in kwargs and param.default is not None:
                    kwargs[param.name] = param.default

            # 검증
            valid, error = self.validate_params(kwargs)
            if not valid:
                return ToolResult(success=False, error=error)

            # 실행
            result = self.func(**kwargs)

            execution_time = (time.time() - start_time) * 1000
            return ToolResult(
                success=True,
                data=result,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Tool execution error: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )


def create_tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    parameters: Optional[List[ToolParameter]] = None,
):
    """
    함수를 도구로 변환하는 데코레이터

    Usage:
        @create_tool(name="search", description="검색 도구")
        def search_documents(query: str, limit: int = 10):
            ...
    """
    def decorator(func: Callable) -> FunctionTool:
        return FunctionTool(
            func=func,
            name=name,
            description=description,
            parameters=parameters,
        )
    return decorator


class ToolRegistry:
    """
    도구 레지스트리

    도구 등록 및 조회
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(
        self,
        tool: Tool,
        category: Optional[str] = None,
    ) -> None:
        """도구 등록"""
        self._tools[tool.name] = tool

        if category:
            if category not in self._categories:
                self._categories[category] = []
            self._categories[category].append(tool.name)

        logger.info(f"Registered tool: {tool.name}")

    def register_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None,
    ) -> FunctionTool:
        """함수를 도구로 등록"""
        tool = FunctionTool(func, name, description)
        self.register(tool, category)
        return tool

    def get(self, name: str) -> Optional[Tool]:
        """도구 조회"""
        return self._tools.get(name)

    def get_all(self) -> List[Tool]:
        """모든 도구"""
        return list(self._tools.values())

    def get_by_category(self, category: str) -> List[Tool]:
        """카테고리별 도구"""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]

    def get_schemas(self, tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """도구 스키마 목록"""
        if tools:
            return [
                self._tools[name].get_schema()
                for name in tools
                if name in self._tools
            ]
        return [tool.get_schema() for tool in self._tools.values()]

    def list_tools(self) -> List[Dict[str, Any]]:
        """도구 목록"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": [p.name for p in tool.parameters],
            }
            for tool in self._tools.values()
        ]

    def unregister(self, name: str) -> bool:
        """도구 등록 해제"""
        if name in self._tools:
            del self._tools[name]
            # 카테고리에서도 제거
            for category in self._categories.values():
                if name in category:
                    category.remove(name)
            return True
        return False


# =============================================================================
# 내장 도구들
# =============================================================================

class SearchTool(Tool):
    """문서 검색 도구"""

    def __init__(self, search_func: Callable):
        super().__init__(
            name="search_documents",
            description="금융 문서에서 관련 정보를 검색합니다.",
            parameters=[
                ToolParameter(
                    name="query",
                    param_type=ParameterType.STRING,
                    description="검색 쿼리",
                    required=True,
                ),
                ToolParameter(
                    name="top_k",
                    param_type=ParameterType.INTEGER,
                    description="반환할 문서 수",
                    required=False,
                    default=5,
                ),
                ToolParameter(
                    name="filters",
                    param_type=ParameterType.OBJECT,
                    description="필터 조건 (company, date_range 등)",
                    required=False,
                ),
            ],
        )
        self.search_func = search_func

    def execute(self, **kwargs) -> ToolResult:
        start_time = time.time()
        try:
            query = kwargs.get("query", "")
            top_k = kwargs.get("top_k", 5)
            filters = kwargs.get("filters", {})

            results = self.search_func(query, top_k, filters)

            return ToolResult(
                success=True,
                data=results,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )


class CalculatorTool(Tool):
    """계산기 도구"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="수학 계산을 수행합니다. 금융 지표 계산에 사용됩니다.",
            parameters=[
                ToolParameter(
                    name="expression",
                    param_type=ParameterType.STRING,
                    description="계산할 수학 표현식 (예: '100 * 1.15', '(500-300)/300*100')",
                    required=True,
                ),
            ],
        )

    def execute(self, **kwargs) -> ToolResult:
        start_time = time.time()
        try:
            expression = kwargs.get("expression", "")

            # 안전한 수학 연산만 허용
            allowed_chars = set("0123456789+-*/.() ")
            if not all(c in allowed_chars for c in expression):
                return ToolResult(
                    success=False,
                    error="Invalid characters in expression",
                )

            result = eval(expression)

            return ToolResult(
                success=True,
                data={"expression": expression, "result": result},
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )


class DataLookupTool(Tool):
    """데이터 조회 도구"""

    def __init__(self, lookup_func: Optional[Callable] = None):
        super().__init__(
            name="data_lookup",
            description="특정 회사나 지표의 데이터를 조회합니다.",
            parameters=[
                ToolParameter(
                    name="entity",
                    param_type=ParameterType.STRING,
                    description="조회할 엔티티 (회사명, 종목코드 등)",
                    required=True,
                ),
                ToolParameter(
                    name="metric",
                    param_type=ParameterType.STRING,
                    description="조회할 지표 (매출, 영업이익, PER 등)",
                    required=True,
                ),
                ToolParameter(
                    name="period",
                    param_type=ParameterType.STRING,
                    description="기간 (2024Q1, 2023, 최근분기 등)",
                    required=False,
                    default="최근",
                ),
            ],
        )
        self.lookup_func = lookup_func or self._default_lookup

    def _default_lookup(
        self,
        entity: str,
        metric: str,
        period: str,
    ) -> Dict[str, Any]:
        """기본 조회 (더미 데이터)"""
        return {
            "entity": entity,
            "metric": metric,
            "period": period,
            "value": None,
            "note": "데이터 조회 함수가 설정되지 않았습니다.",
        }

    def execute(self, **kwargs) -> ToolResult:
        start_time = time.time()
        try:
            entity = kwargs.get("entity", "")
            metric = kwargs.get("metric", "")
            period = kwargs.get("period", "최근")

            result = self.lookup_func(entity, metric, period)

            return ToolResult(
                success=True,
                data=result,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )


class ComparisonTool(Tool):
    """비교 분석 도구"""

    def __init__(self):
        super().__init__(
            name="compare",
            description="두 개 이상의 항목을 비교 분석합니다.",
            parameters=[
                ToolParameter(
                    name="items",
                    param_type=ParameterType.ARRAY,
                    description="비교할 항목들",
                    required=True,
                    items_type=ParameterType.STRING,
                ),
                ToolParameter(
                    name="aspects",
                    param_type=ParameterType.ARRAY,
                    description="비교 관점들 (매출, 성장률 등)",
                    required=False,
                    items_type=ParameterType.STRING,
                ),
            ],
        )

    def execute(self, **kwargs) -> ToolResult:
        start_time = time.time()
        try:
            items = kwargs.get("items", [])
            aspects = kwargs.get("aspects", ["전반적"])

            if len(items) < 2:
                return ToolResult(
                    success=False,
                    error="비교를 위해 최소 2개 항목이 필요합니다.",
                )

            result = {
                "items": items,
                "aspects": aspects,
                "comparison": {
                    "note": "비교 분석 결과가 생성됩니다.",
                },
            }

            return ToolResult(
                success=True,
                data=result,
                execution_time_ms=(time.time() - start_time) * 1000,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
