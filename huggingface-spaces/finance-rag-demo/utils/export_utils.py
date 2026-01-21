# -*- coding: utf-8 -*-
"""
Export Utilities

PDF/CSV 내보내기 유틸리티
"""

import csv
import io
from typing import List, Dict, Any
from datetime import datetime


def export_to_csv(sessions: List[Dict[str, Any]]) -> str:
    """대화 히스토리를 CSV로 내보내기"""
    output = io.StringIO()
    writer = csv.writer(output)

    # 헤더
    writer.writerow([
        "Timestamp",
        "Question",
        "Answer",
        "Search Mode",
        "Response Time (s)",
        "Sources Count"
    ])

    # 데이터
    for session in sessions:
        timestamp = session.get("timestamp", "")
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        writer.writerow([
            timestamp,
            session.get("question", ""),
            session.get("answer", "").replace("\n", " "),
            session.get("search_mode", "hybrid"),
            f"{session.get('elapsed_time', 0):.2f}",
            len(session.get("sources", []))
        ])

    return output.getvalue()


def export_sources_to_csv(sources: List[Dict[str, Any]]) -> str:
    """검색 소스를 CSV로 내보내기"""
    output = io.StringIO()
    writer = csv.writer(output)

    # 헤더
    writer.writerow([
        "Title",
        "Category",
        "Company",
        "Date",
        "Source",
        "Score",
        "Content Preview"
    ])

    # 데이터
    for source in sources:
        meta = source.get("metadata", {})
        writer.writerow([
            meta.get("title", "N/A"),
            meta.get("category", "N/A"),
            meta.get("company", "N/A"),
            meta.get("date", "N/A"),
            meta.get("source", "N/A"),
            f"{source.get('score', 0):.2%}",
            source.get("content", "")[:200].replace("\n", " ")
        ])

    return output.getvalue()


def export_to_pdf(sessions: List[Dict[str, Any]]) -> bytes:
    """대화 히스토리를 PDF로 내보내기"""
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("fpdf2 패키지가 필요합니다: pip install fpdf2")

    # PDF 생성
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 한글 폰트 설정 (기본 폰트 사용 - 한글 지원 제한)
    # HuggingFace Spaces에서는 시스템 폰트 사용 불가하므로 기본 설정 유지
    pdf.add_page()

    # 제목
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 15, "Finance RAG Pro - Chat History", ln=True, align="C")

    # 날짜
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 10, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)

    # 대화 내역
    for i, session in enumerate(sessions, 1):
        # 세션 헤더
        pdf.set_font("Helvetica", "B", 12)
        timestamp = session.get("timestamp", "")
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass

        pdf.set_fill_color(240, 240, 240)
        pdf.cell(0, 8, f"[{i}] {timestamp} | {session.get('search_mode', 'hybrid').upper()}", ln=True, fill=True)
        pdf.ln(3)

        # 질문
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Question:", ln=True)
        pdf.set_font("Helvetica", "", 10)

        # 질문 텍스트 (ASCII로 변환)
        question = session.get("question", "")
        # 한글 등 비ASCII 문자를 안전하게 처리
        safe_question = question.encode('latin-1', errors='replace').decode('latin-1')
        pdf.multi_cell(0, 5, safe_question)
        pdf.ln(3)

        # 답변
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Answer:", ln=True)
        pdf.set_font("Helvetica", "", 10)

        # 답변 텍스트 (제한 + ASCII 변환)
        answer = session.get("answer", "")[:1000]
        safe_answer = answer.encode('latin-1', errors='replace').decode('latin-1')
        pdf.multi_cell(0, 5, safe_answer)

        # 메타 정보
        pdf.set_font("Helvetica", "I", 8)
        elapsed = session.get("elapsed_time", 0)
        sources_count = len(session.get("sources", []))
        pdf.cell(0, 5, f"Response: {elapsed:.2f}s | Sources: {sources_count}", ln=True)
        pdf.ln(8)

        # 페이지 넘침 방지
        if pdf.get_y() > 260:
            pdf.add_page()

    return pdf.output()


def create_session_summary(session: Dict[str, Any]) -> str:
    """세션 요약 텍스트 생성"""
    question = session.get("question", "")[:50]
    answer_preview = session.get("answer", "")[:100]
    timestamp = session.get("timestamp", "")

    if isinstance(timestamp, str):
        try:
            dt = datetime.fromisoformat(timestamp)
            timestamp = dt.strftime("%m/%d %H:%M")
        except Exception:
            timestamp = "N/A"

    return f"[{timestamp}] Q: {question}... → {answer_preview}..."
