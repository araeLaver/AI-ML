# -*- coding: utf-8 -*-
"""
Finance RAG - 관리자 대시보드

시스템 모니터링, 성능 분석, 보안 감사 대시보드
"""

import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import time
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title="Finance RAG Admin",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS 스타일
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-dark: #0e1117;
    --bg-card: #1e2530;
    --text-primary: #fafafa;
    --text-secondary: #8b949e;
    --accent: #58a6ff;
    --success: #3fb950;
    --warning: #d29922;
    --error: #f85149;
    --border: #30363d;
}

* { font-family: 'Inter', sans-serif; }

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.metric-title {
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
}

.metric-change {
    font-size: 0.85rem;
    margin-top: 0.25rem;
}

.metric-change.positive { color: var(--success); }
.metric-change.negative { color: var(--error); }
.metric-change.neutral { color: var(--text-secondary); }

.status-badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}

.status-ok { background: rgba(63, 185, 80, 0.2); color: var(--success); }
.status-warning { background: rgba(210, 153, 34, 0.2); color: var(--warning); }
.status-error { background: rgba(248, 81, 73, 0.2); color: var(--error); }

.section-header {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

.log-entry {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-family: monospace;
    font-size: 0.85rem;
}

.log-time { color: var(--text-secondary); }
.log-action { color: var(--accent); font-weight: 600; }
.log-user { color: var(--warning); }
.log-success { color: var(--success); }
.log-failure { color: var(--error); }

.chart-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# API 클라이언트
# ============================================================
import requests
from typing import Optional

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class DashboardAPI:
    """대시보드 API 클라이언트"""

    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _get(self, endpoint: str) -> Optional[dict]:
        try:
            response = self.session.get(f"{self.base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"API Error: {e}")
        return None

    def _post(self, endpoint: str, data: dict = None) -> Optional[dict]:
        try:
            response = self.session.post(
                f"{self.base_url}{endpoint}",
                json=data or {},
                timeout=10
            )
            return response.json()
        except Exception as e:
            st.error(f"API Error: {e}")
        return None

    def get_health(self) -> dict:
        return self._get("/api/v1/health") or {"status": "unknown"}

    def get_stats(self) -> dict:
        return self._get("/api/v1/stats") or {}

    def get_cache_stats(self) -> dict:
        return self._get("/api/v1/performance/cache/stats") or {}

    def get_performance_metrics(self) -> dict:
        return self._get("/api/v1/performance/metrics") or {}

    def get_security_health(self) -> dict:
        return self._get("/api/v1/security/health") or {}

    def get_audit_logs(self, limit: int = 50) -> list:
        result = self._get(f"/api/v1/security/audit/logs?limit={limit}")
        return result if isinstance(result, list) else []

    def get_audit_stats(self) -> dict:
        return self._get("/api/v1/security/audit/stats") or {}


@st.cache_resource
def get_api_client():
    return DashboardAPI()


# ============================================================
# 메인 앱
# ============================================================

def main():
    api = get_api_client()

    # 사이드바
    with st.sidebar:
        st.markdown("## Admin Dashboard")
        st.markdown("---")

        page = st.radio(
            "Navigation",
            ["Overview", "Performance", "Security", "Logs", "Settings"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # 자동 새로고침
        auto_refresh = st.checkbox("Auto Refresh", value=False)
        if auto_refresh:
            refresh_interval = st.slider("Interval (sec)", 5, 60, 30)
            st.markdown(f"*Refreshing every {refresh_interval}s*")

        st.markdown("---")
        st.markdown(f"**API URL:** `{API_BASE_URL}`")

        if st.button("Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # 페이지 라우팅
    if page == "Overview":
        show_overview_page(api)
    elif page == "Performance":
        show_performance_page(api)
    elif page == "Security":
        show_security_page(api)
    elif page == "Logs":
        show_logs_page(api)
    elif page == "Settings":
        show_settings_page(api)

    # 자동 새로고침
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


# ============================================================
# Overview 페이지
# ============================================================
def show_overview_page(api: DashboardAPI):
    st.title("System Overview")

    # 상태 요약
    col1, col2, col3, col4 = st.columns(4)

    # API 상태
    health = api.get_health()
    with col1:
        status = health.get("status", "unknown")
        status_class = "ok" if status == "ok" else ("warning" if status == "degraded" else "error")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">API Status</div>
            <span class="status-badge status-{status_class}">{status.upper()}</span>
        </div>
        """, unsafe_allow_html=True)

    # 문서 수
    stats = api.get_stats()
    with col2:
        doc_count = stats.get("total_documents", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Documents</div>
            <div class="metric-value">{doc_count:,}</div>
        </div>
        """, unsafe_allow_html=True)

    # 캐시 적중률
    cache_stats = api.get_cache_stats()
    with col3:
        hit_rate = cache_stats.get("stats", {}).get("hit_rate", 0) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Cache Hit Rate</div>
            <div class="metric-value">{hit_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # 보안 상태
    sec_health = api.get_security_health()
    with col4:
        jwt_status = "ON" if sec_health.get("jwt_available", False) else "OFF"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">JWT Auth</div>
            <div class="metric-value">{jwt_status}</div>
        </div>
        """, unsafe_allow_html=True)

    # 성능 요약
    st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)

    metrics = api.get_performance_metrics()
    if metrics.get("summary"):
        summary = metrics["summary"]

        perf_cols = st.columns(4)
        with perf_cols[0]:
            st.metric(
                "Total Operations",
                f"{summary.get('total_operations', 0):,}"
            )
        with perf_cols[1]:
            st.metric(
                "Avg Response Time",
                f"{summary.get('avg_response_time', 0):.0f}ms"
            )
        with perf_cols[2]:
            st.metric(
                "Peak Response Time",
                f"{summary.get('max_response_time', 0):.0f}ms"
            )
        with perf_cols[3]:
            st.metric(
                "Error Rate",
                f"{summary.get('error_rate', 0):.2f}%"
            )

    # 시스템 기능
    st.markdown('<div class="section-header">System Features</div>', unsafe_allow_html=True)

    feature_cols = st.columns(3)

    features = {
        "RAG Engine": health.get("llm_available", False),
        "Vector Store": health.get("vectorstore_available", False),
        "WebSocket": True,
        "SSE Streaming": True,
        "Caching": cache_stats.get("enabled", False),
        "Rate Limiting": sec_health.get("features", {}).get("rbac", False),
    }

    for i, (feature, enabled) in enumerate(features.items()):
        with feature_cols[i % 3]:
            status = "ok" if enabled else "error"
            icon = "" if enabled else ""
            st.markdown(f"""
            <div class="metric-card" style="padding: 1rem;">
                <span class="status-badge status-{status}">{icon} {feature}</span>
            </div>
            """, unsafe_allow_html=True)

    # 최근 활동
    st.markdown('<div class="section-header">Recent Activity</div>', unsafe_allow_html=True)

    audit_logs = api.get_audit_logs(limit=5)
    if audit_logs:
        for log in audit_logs[:5]:
            timestamp = log.get("timestamp", "")[:19].replace("T", " ")
            action = log.get("action", "unknown")
            user = log.get("username", "anonymous")
            success = log.get("success", True)

            success_class = "log-success" if success else "log-failure"
            success_text = "SUCCESS" if success else "FAILED"

            st.markdown(f"""
            <div class="log-entry">
                <span class="log-time">{timestamp}</span> |
                <span class="log-action">{action}</span> |
                <span class="log-user">{user}</span> |
                <span class="{success_class}">{success_text}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent activity logs")


# ============================================================
# Performance 페이지
# ============================================================
def show_performance_page(api: DashboardAPI):
    st.title("Performance Monitoring")

    # 캐시 통계
    st.markdown('<div class="section-header">Cache Statistics</div>', unsafe_allow_html=True)

    cache_stats = api.get_cache_stats()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Cache Overview")
        stats = cache_stats.get("stats", {})

        cache_cols = st.columns(4)
        with cache_cols[0]:
            st.metric("Hits", f"{stats.get('hits', 0):,}")
        with cache_cols[1]:
            st.metric("Misses", f"{stats.get('misses', 0):,}")
        with cache_cols[2]:
            st.metric("Sets", f"{stats.get('sets', 0):,}")
        with cache_cols[3]:
            st.metric("Deletes", f"{stats.get('deletes', 0):,}")

        # 캐시 적중률 차트
        if stats.get("hits", 0) + stats.get("misses", 0) > 0:
            import plotly.graph_objects as go

            fig = go.Figure(data=[go.Pie(
                labels=["Hits", "Misses"],
                values=[stats.get("hits", 0), stats.get("misses", 0)],
                hole=0.6,
                marker_colors=["#3fb950", "#f85149"]
            )])
            fig.update_layout(
                showlegend=True,
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa")
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Cache Configuration")
        st.json({
            "Backend": cache_stats.get("backend", "unknown"),
            "Enabled": cache_stats.get("enabled", False),
            "Prefix": cache_stats.get("prefix", "N/A"),
            "Default TTL": f"{cache_stats.get('default_ttl', 0)}s"
        })

    # 성능 메트릭
    st.markdown('<div class="section-header">Performance Metrics</div>', unsafe_allow_html=True)

    metrics = api.get_performance_metrics()
    if metrics.get("metrics"):
        op_metrics = metrics["metrics"]

        st.dataframe(
            [
                {
                    "Operation": op,
                    "Count": data.get("count", 0),
                    "Avg Time (ms)": f"{data.get('avg_time', 0) * 1000:.2f}",
                    "Min Time (ms)": f"{data.get('min_time', 0) * 1000:.2f}",
                    "Max Time (ms)": f"{data.get('max_time', 0) * 1000:.2f}",
                }
                for op, data in op_metrics.items()
            ],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No performance metrics available yet")

    # 캐시 관리
    st.markdown('<div class="section-header">Cache Management</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pattern = st.text_input("Clear Pattern", placeholder="e.g., query:*")
        if st.button("Clear Cache", type="primary"):
            result = api._post("/api/v1/performance/cache/clear", {"pattern": pattern})
            if result and result.get("success"):
                st.success(f"Cleared {result.get('cleared_count', 0)} cache entries")
            else:
                st.error("Failed to clear cache")

    with col2:
        if st.button("Reset Metrics"):
            result = api._post("/api/v1/performance/metrics/reset")
            if result and result.get("success"):
                st.success("Metrics reset successfully")
            else:
                st.error("Failed to reset metrics")


# ============================================================
# Security 페이지
# ============================================================
def show_security_page(api: DashboardAPI):
    st.title("Security Dashboard")

    # 보안 상태
    st.markdown('<div class="section-header">Security Status</div>', unsafe_allow_html=True)

    sec_health = api.get_security_health()
    features = sec_health.get("features", {})

    feature_cols = st.columns(4)
    feature_items = [
        ("JWT Auth", features.get("jwt_auth", False)),
        ("API Key Auth", features.get("api_key_auth", False)),
        ("RBAC", features.get("rbac", False)),
        ("Audit Logging", features.get("audit_logging", False)),
    ]

    for i, (name, enabled) in enumerate(feature_items):
        with feature_cols[i]:
            status = "ok" if enabled else "warning"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{name}</div>
                <span class="status-badge status-{status}">{'ENABLED' if enabled else 'DISABLED'}</span>
            </div>
            """, unsafe_allow_html=True)

    # 감사 통계
    st.markdown('<div class="section-header">Audit Statistics</div>', unsafe_allow_html=True)

    audit_stats = api.get_audit_stats()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Audit Logs", f"{audit_stats.get('total_logs', 0):,}")

        if audit_stats.get("by_action"):
            import plotly.graph_objects as go

            actions = list(audit_stats["by_action"].keys())
            counts = list(audit_stats["by_action"].values())

            fig = go.Figure(data=[go.Bar(
                x=actions,
                y=counts,
                marker_color="#58a6ff"
            )])
            fig.update_layout(
                title="Actions Distribution",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#fafafa"),
                xaxis=dict(gridcolor="#30363d"),
                yaxis=dict(gridcolor="#30363d")
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if audit_stats.get("by_user"):
            st.markdown("### Activity by User")
            for user, count in list(audit_stats["by_user"].items())[:10]:
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #30363d;">
                    <span>{user}</span>
                    <span style="font-weight: 600;">{count}</span>
                </div>
                """, unsafe_allow_html=True)


# ============================================================
# Logs 페이지
# ============================================================
def show_logs_page(api: DashboardAPI):
    st.title("Audit Logs")

    # 필터
    col1, col2, col3 = st.columns(3)
    with col1:
        action_filter = st.selectbox(
            "Action Filter",
            ["All", "login", "logout", "query_execute", "document_create", "api_key_create"]
        )
    with col2:
        limit = st.selectbox("Show", [25, 50, 100, 200], index=1)
    with col3:
        if st.button("Refresh", use_container_width=True):
            st.rerun()

    # 로그 목록
    logs = api.get_audit_logs(limit=limit)

    if logs:
        for log in logs:
            if action_filter != "All" and log.get("action") != action_filter:
                continue

            timestamp = log.get("timestamp", "")[:19].replace("T", " ")
            action = log.get("action", "unknown")
            user = log.get("username", "anonymous")
            user_id = log.get("user_id", "-")
            resource = log.get("resource", "-")
            success = log.get("success", True)

            success_class = "log-success" if success else "log-failure"
            success_text = "SUCCESS" if success else "FAILED"

            st.markdown(f"""
            <div class="log-entry">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span class="log-time">{timestamp}</span>
                    <span class="{success_class}">{success_text}</span>
                </div>
                <div>
                    <span class="log-action">{action}</span> |
                    User: <span class="log-user">{user}</span> ({user_id}) |
                    Resource: {resource}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No audit logs available")


# ============================================================
# Settings 페이지
# ============================================================
def show_settings_page(api: DashboardAPI):
    st.title("System Settings")

    # API 설정
    st.markdown('<div class="section-header">API Configuration</div>', unsafe_allow_html=True)

    config = api._get("/api/v1/performance/config")
    if config:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Cache Settings")
            st.json(config.get("cache", {}))

        with col2:
            st.markdown("### Batch Settings")
            st.json(config.get("batch", {}))

    # 시스템 정보
    st.markdown('<div class="section-header">System Information</div>', unsafe_allow_html=True)

    root_info = api._get("/")
    if root_info:
        st.json({
            "Name": root_info.get("name"),
            "Version": root_info.get("version"),
            "Description": root_info.get("description")
        })

    # 엔드포인트 목록
    st.markdown('<div class="section-header">Available Endpoints</div>', unsafe_allow_html=True)

    if root_info:
        for category, endpoints in root_info.items():
            if isinstance(endpoints, dict) and category not in ["name", "version", "description", "docs", "health"]:
                with st.expander(category.title()):
                    for name, path in endpoints.items():
                        st.code(f"{name}: {path}")


if __name__ == "__main__":
    main()
