# Function Calling Tools
"""
LLM Function Calling 도구 정의 (Step 2: Tool Use)
"""

from typing import Dict, Any, List, Callable
import json

# OpenAI Function Calling 형식의 도구 정의
FINANCIAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_transaction",
            "description": "금융 거래를 분석하여 이상거래 여부를 판단합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "분석할 거래의 ID"
                    },
                    "amount": {
                        "type": "number",
                        "description": "거래 금액 (원)"
                    },
                    "timestamp": {
                        "type": "string",
                        "description": "거래 시간 (ISO 형식)"
                    },
                    "location": {
                        "type": "string",
                        "description": "거래 위치"
                    },
                    "is_international": {
                        "type": "boolean",
                        "description": "해외 거래 여부"
                    }
                },
                "required": ["transaction_id", "amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_account_history",
            "description": "고객의 최근 거래 내역을 조회합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "계좌 번호"
                    },
                    "days": {
                        "type": "integer",
                        "description": "조회할 기간 (일)",
                        "default": 30
                    }
                },
                "required": ["account_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_score",
            "description": "특정 거래 또는 계좌의 위험 점수를 계산합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "거래 ID"
                    },
                    "include_history": {
                        "type": "boolean",
                        "description": "과거 거래 이력 포함 여부",
                        "default": True
                    }
                },
                "required": ["transaction_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "block_transaction",
            "description": "의심 거래를 차단합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "차단할 거래 ID"
                    },
                    "reason": {
                        "type": "string",
                        "description": "차단 사유"
                    }
                },
                "required": ["transaction_id", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_alert",
            "description": "담당자에게 알림을 전송합니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "alert_type": {
                        "type": "string",
                        "enum": ["fraud_suspected", "high_risk", "unusual_pattern"],
                        "description": "알림 유형"
                    },
                    "transaction_id": {
                        "type": "string",
                        "description": "관련 거래 ID"
                    },
                    "message": {
                        "type": "string",
                        "description": "알림 메시지"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "우선순위"
                    }
                },
                "required": ["alert_type", "transaction_id", "message"]
            }
        }
    }
]


class ToolExecutor:
    """
    Function Calling 도구 실행기

    Usage:
        executor = ToolExecutor()
        result = executor.execute("analyze_transaction", {"transaction_id": "TXN001", "amount": 1000000})
    """

    def __init__(self):
        self._handlers: Dict[str, Callable] = {
            "analyze_transaction": self._analyze_transaction,
            "get_account_history": self._get_account_history,
            "get_risk_score": self._get_risk_score,
            "block_transaction": self._block_transaction,
            "send_alert": self._send_alert,
        }

    def execute(self, function_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        도구 실행

        Args:
            function_name: 함수 이름
            arguments: 함수 인자

        Returns:
            실행 결과
        """
        if function_name not in self._handlers:
            return {"error": f"Unknown function: {function_name}"}

        handler = self._handlers[function_name]
        return handler(**arguments)

    def execute_from_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """LLM 도구 호출 결과로부터 실행"""
        function_name = tool_call["function"]["name"]
        arguments = tool_call["function"]["arguments"]

        # 문자열인 경우 JSON 파싱
        if isinstance(arguments, str):
            arguments = json.loads(arguments)

        return self.execute(function_name, arguments)

    # --- 도구 핸들러 구현 ---

    def _analyze_transaction(
        self,
        transaction_id: str,
        amount: float,
        timestamp: str = None,
        location: str = None,
        is_international: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """거래 분석 (샘플 구현)"""
        # 실제로는 ML 모델 호출
        risk_score = 0

        # 간단한 규칙 기반 분석
        if amount > 10000000:  # 1천만원 초과
            risk_score += 30
        if amount > 50000000:  # 5천만원 초과
            risk_score += 20
        if is_international:
            risk_score += 20
        if location and "해외" in location:
            risk_score += 15

        risk_level = "low"
        if risk_score >= 30:
            risk_level = "medium"
        if risk_score >= 50:
            risk_level = "high"
        if risk_score >= 70:
            risk_level = "critical"

        return {
            "transaction_id": transaction_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "is_suspicious": risk_score >= 50,
            "analysis_details": {
                "amount_risk": "high" if amount > 10000000 else "low",
                "international_risk": "yes" if is_international else "no",
                "location_risk": location if location else "unknown"
            }
        }

    def _get_account_history(
        self,
        account_id: str,
        days: int = 30,
        **kwargs
    ) -> Dict[str, Any]:
        """계좌 거래 내역 조회 (샘플 구현)"""
        # 실제로는 DB 조회
        return {
            "account_id": account_id,
            "period_days": days,
            "total_transactions": 45,
            "total_amount": 15000000,
            "average_amount": 333333,
            "max_amount": 2000000,
            "international_count": 2,
            "recent_transactions": [
                {"date": "2024-01-15", "amount": 500000, "type": "출금"},
                {"date": "2024-01-14", "amount": 1200000, "type": "입금"},
                {"date": "2024-01-13", "amount": 50000, "type": "출금"},
            ]
        }

    def _get_risk_score(
        self,
        transaction_id: str,
        include_history: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """위험 점수 계산 (샘플 구현)"""
        # 실제로는 ML 모델 + 규칙 엔진 조합
        import random
        random.seed(hash(transaction_id) % 100)

        score = random.randint(10, 90)

        return {
            "transaction_id": transaction_id,
            "risk_score": score,
            "confidence": 0.85,
            "factors": {
                "amount_factor": random.uniform(0.1, 0.4),
                "time_factor": random.uniform(0.05, 0.2),
                "location_factor": random.uniform(0.05, 0.3),
                "history_factor": random.uniform(0.1, 0.3) if include_history else 0
            },
            "recommendation": "block" if score > 70 else "review" if score > 40 else "approve"
        }

    def _block_transaction(
        self,
        transaction_id: str,
        reason: str,
        **kwargs
    ) -> Dict[str, Any]:
        """거래 차단 (샘플 구현)"""
        # 실제로는 거래 시스템 API 호출
        return {
            "transaction_id": transaction_id,
            "status": "blocked",
            "reason": reason,
            "blocked_at": "2024-01-15T10:30:00Z",
            "blocked_by": "fraud_detection_system",
            "next_action": "manual_review_required"
        }

    def _send_alert(
        self,
        alert_type: str,
        transaction_id: str,
        message: str,
        priority: str = "medium",
        **kwargs
    ) -> Dict[str, Any]:
        """알림 전송 (샘플 구현)"""
        # 실제로는 알림 서비스 API 호출
        return {
            "alert_id": f"ALERT_{transaction_id}_{alert_type}",
            "status": "sent",
            "recipients": ["fraud_team@bank.com", "risk_manager@bank.com"],
            "alert_type": alert_type,
            "priority": priority,
            "message": message,
            "sent_at": "2024-01-15T10:30:00Z"
        }


if __name__ == "__main__":
    # 테스트
    executor = ToolExecutor()

    # 거래 분석 테스트
    print("=" * 50)
    print("Transaction Analysis:")
    result = executor.execute("analyze_transaction", {
        "transaction_id": "TXN001",
        "amount": 50000000,
        "is_international": True
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 위험 점수 테스트
    print("\n" + "=" * 50)
    print("Risk Score:")
    result = executor.execute("get_risk_score", {
        "transaction_id": "TXN001"
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))
