"""
DART 공시 데이터 수집기

DART Open API를 사용하여 금융 공시 데이터를 수집합니다.
- 공시 목록 검색
- 공시 원문 다운로드
- 텍스트 추출 및 정제

API 키 발급: https://opendart.fss.or.kr/
"""

import os
import re
import json
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, asdict
from io import BytesIO

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Disclosure:
    """공시 정보"""
    rcept_no: str  # 접수번호
    corp_code: str  # 고유번호
    corp_name: str  # 회사명
    stock_code: str  # 종목코드
    corp_cls: str  # 법인구분 (Y:유가, K:코스닥, N:코넥스, E:기타)
    report_nm: str  # 보고서명
    rcept_dt: str  # 접수일자
    flr_nm: str  # 공시제출인명
    rm: str  # 비고
    content: Optional[str] = None  # 공시 내용


class DARTCollector:
    """DART 공시 데이터 수집기"""

    BASE_URL = "https://opendart.fss.or.kr/api"

    # 주요 공시 유형 (RAG에 유용한 것들)
    IMPORTANT_REPORT_TYPES = [
        "사업보고서",
        "분기보고서",
        "반기보고서",
        "주요사항보고서",
        "공정공시",
        "실적",
        "영업실적",
        "매출액",
        "수주",
        "계약",
        "투자",
        "유상증자",
        "무상증자",
        "합병",
        "분할",
        "배당",
    ]

    # 주요 기업 (시가총액 상위)
    MAJOR_CORPS = {
        "삼성전자": "00126380",
        "SK하이닉스": "00164779",
        "LG에너지솔루션": "01651730",
        "삼성바이오로직스": "00771305",
        "삼성SDI": "00126186",
        "현대차": "00164742",
        "기아": "00164715",
        "셀트리온": "00421045",
        "POSCO홀딩스": "00117631",
        "KB금융": "00736911",
        "신한지주": "00382199",
        "NAVER": "00401731",
        "카카오": "00918444",
        "하나금융지주": "00547583",
        "LG화학": "00164497",
        "현대모비스": "00164788",
        "삼성물산": "00126236",
        "삼성생명": "00623327",
        "KT&G": "00140042",
        "SK이노베이션": "00631518",
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: DART API 키 (없으면 환경변수에서 읽음)
        """
        self.api_key = api_key or os.getenv("DART_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DART API 키가 필요합니다. "
                "https://opendart.fss.or.kr에서 발급받아 "
                "DART_API_KEY 환경변수로 설정하세요."
            )

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Finance RAG API)"
        })

        # 저장 경로
        self.data_dir = Path(__file__).parent.parent.parent / "data" / "dart"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _request(self, endpoint: str, params: dict) -> dict:
        """API 요청"""
        params["crtfc_key"] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()

        # Rate limiting (초당 2회 제한)
        time.sleep(0.5)

        return response.json()

    def _request_binary(self, endpoint: str, params: dict) -> bytes:
        """바이너리 응답 API 요청 (문서 다운로드용)"""
        params["crtfc_key"] = self.api_key

        url = f"{self.BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params, timeout=60)
        response.raise_for_status()

        time.sleep(0.5)

        return response.content

    def search_disclosures(
        self,
        corp_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        report_type: str = "A",  # A:정기공시, B:주요사항, C:발행공시, D:지분공시, E:기타, F:외부감사, G:펀드, H:자산유동화, I:거래소공시, J:공정위공시
        page_no: int = 1,
        page_count: int = 100,
    ) -> list[Disclosure]:
        """
        공시 목록 검색

        Args:
            corp_code: 고유번호 (특정 기업만 검색)
            start_date: 시작일 (YYYYMMDD)
            end_date: 종료일 (YYYYMMDD)
            report_type: 공시유형
            page_no: 페이지 번호
            page_count: 페이지당 건수 (최대 100)

        Returns:
            공시 목록
        """
        params = {
            "page_no": page_no,
            "page_count": page_count,
        }

        if corp_code:
            params["corp_code"] = corp_code
        if start_date:
            params["bgn_de"] = start_date
        if end_date:
            params["end_de"] = end_date
        if report_type:
            params["pblntf_ty"] = report_type

        result = self._request("list.json", params)

        if result.get("status") != "000":
            if result.get("status") == "013":
                # 조회된 데이터 없음
                return []
            raise Exception(f"DART API 오류: {result.get('message')}")

        disclosures = []
        for item in result.get("list", []):
            disclosures.append(Disclosure(
                rcept_no=item.get("rcept_no", ""),
                corp_code=item.get("corp_code", ""),
                corp_name=item.get("corp_name", ""),
                stock_code=item.get("stock_code", ""),
                corp_cls=item.get("corp_cls", ""),
                report_nm=item.get("report_nm", ""),
                rcept_dt=item.get("rcept_dt", ""),
                flr_nm=item.get("flr_nm", ""),
                rm=item.get("rm", ""),
            ))

        return disclosures

    def download_document(self, rcept_no: str) -> Optional[str]:
        """
        공시 원문 다운로드 및 텍스트 추출

        Args:
            rcept_no: 접수번호

        Returns:
            추출된 텍스트 (실패 시 None)
        """
        try:
            # ZIP 파일 다운로드
            content = self._request_binary("document.xml", {"rcept_no": rcept_no})

            # ZIP 압축 해제
            with zipfile.ZipFile(BytesIO(content)) as zf:
                text_parts = []

                for filename in zf.namelist():
                    if filename.endswith(".xml"):
                        with zf.open(filename) as f:
                            xml_content = f.read().decode("utf-8", errors="ignore")
                            text = self._extract_text_from_xml(xml_content)
                            if text:
                                text_parts.append(text)

                return "\n\n".join(text_parts) if text_parts else None

        except zipfile.BadZipFile:
            # ZIP이 아닌 경우 (XML 직접 반환)
            try:
                text = content.decode("utf-8", errors="ignore")
                return self._extract_text_from_xml(text)
            except Exception:
                return None
        except Exception as e:
            print(f"문서 다운로드 실패 ({rcept_no}): {e}")
            return None

    def _extract_text_from_xml(self, xml_content: str) -> str:
        """XML에서 텍스트 추출"""
        # HTML 태그 제거
        text = re.sub(r"<[^>]+>", " ", xml_content)

        # 특수문자 정리
        text = re.sub(r"&[a-zA-Z]+;", " ", text)
        text = re.sub(r"&#\d+;", " ", text)

        # 공백 정리
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        # 너무 짧은 텍스트 필터링
        if len(text) < 100:
            return ""

        return text

    def collect_major_corps(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        download_content: bool = True,
        max_per_corp: int = 50,
    ) -> list[Disclosure]:
        """
        주요 기업 공시 수집

        Args:
            start_date: 시작일 (기본: 1년 전)
            end_date: 종료일 (기본: 오늘)
            download_content: 원문 다운로드 여부
            max_per_corp: 기업당 최대 수집 건수

        Returns:
            수집된 공시 목록
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        all_disclosures = []

        for corp_name, corp_code in self.MAJOR_CORPS.items():
            print(f"\n[{corp_name}] 공시 수집 중...")

            try:
                disclosures = self.search_disclosures(
                    corp_code=corp_code,
                    start_date=start_date,
                    end_date=end_date,
                    page_count=100,
                )

                # 중요 공시만 필터링
                important = [
                    d for d in disclosures
                    if any(rt in d.report_nm for rt in self.IMPORTANT_REPORT_TYPES)
                ]

                # 최대 건수 제한
                important = important[:max_per_corp]

                print(f"  - 전체: {len(disclosures)}건, 중요: {len(important)}건")

                # 원문 다운로드
                if download_content:
                    for i, disc in enumerate(important):
                        print(f"  - [{i+1}/{len(important)}] {disc.report_nm} 다운로드 중...")
                        disc.content = self.download_document(disc.rcept_no)

                all_disclosures.extend(important)

            except Exception as e:
                print(f"  - 오류: {e}")
                continue

        return all_disclosures

    def collect_by_report_type(
        self,
        report_type: str = "A",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        download_content: bool = True,
        max_count: int = 500,
    ) -> list[Disclosure]:
        """
        공시 유형별 수집

        Args:
            report_type: A(정기), B(주요사항), C(발행), D(지분), E(기타), I(거래소)
            start_date: 시작일
            end_date: 종료일
            download_content: 원문 다운로드 여부
            max_count: 최대 수집 건수

        Returns:
            수집된 공시 목록
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        all_disclosures = []
        page = 1

        while len(all_disclosures) < max_count:
            print(f"\n페이지 {page} 수집 중...")

            disclosures = self.search_disclosures(
                start_date=start_date,
                end_date=end_date,
                report_type=report_type,
                page_no=page,
                page_count=100,
            )

            if not disclosures:
                break

            # 중요 공시 필터링
            important = [
                d for d in disclosures
                if any(rt in d.report_nm for rt in self.IMPORTANT_REPORT_TYPES)
            ]

            print(f"  - 수집: {len(disclosures)}건, 중요: {len(important)}건")

            # 원문 다운로드
            if download_content:
                for i, disc in enumerate(important):
                    if len(all_disclosures) + i >= max_count:
                        break
                    print(f"  - [{len(all_disclosures)+i+1}/{max_count}] {disc.corp_name} - {disc.report_nm}")
                    disc.content = self.download_document(disc.rcept_no)

            all_disclosures.extend(important[:max_count - len(all_disclosures)])

            if len(disclosures) < 100:
                break

            page += 1

        return all_disclosures

    def save_disclosures(
        self,
        disclosures: list[Disclosure],
        filename: str = "disclosures.json",
    ) -> Path:
        """
        공시 데이터 저장

        Args:
            disclosures: 공시 목록
            filename: 저장 파일명

        Returns:
            저장된 파일 경로
        """
        filepath = self.data_dir / filename

        data = {
            "collected_at": datetime.now().isoformat(),
            "total_count": len(disclosures),
            "disclosures": [asdict(d) for d in disclosures],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n저장 완료: {filepath}")
        print(f"총 {len(disclosures)}건")

        # 통계 출력
        corps = set(d.corp_name for d in disclosures)
        with_content = sum(1 for d in disclosures if d.content)
        print(f"기업 수: {len(corps)}")
        print(f"원문 포함: {with_content}건")

        return filepath

    def export_for_rag(
        self,
        disclosures: list[Disclosure],
        output_dir: Optional[Path] = None,
    ) -> list[dict]:
        """
        RAG 시스템용으로 내보내기

        Args:
            disclosures: 공시 목록
            output_dir: 출력 디렉토리

        Returns:
            RAG용 문서 목록
        """
        if not output_dir:
            output_dir = self.data_dir / "rag_documents"
        output_dir.mkdir(parents=True, exist_ok=True)

        documents = []

        for disc in disclosures:
            if not disc.content:
                continue

            doc = {
                "id": disc.rcept_no,
                "content": disc.content,
                "metadata": {
                    "source": f"DART_{disc.rcept_no}",
                    "corp_name": disc.corp_name,
                    "corp_code": disc.corp_code,
                    "stock_code": disc.stock_code,
                    "report_name": disc.report_nm,
                    "report_date": disc.rcept_dt,
                    "type": "dart_disclosure",
                },
            }
            documents.append(doc)

            # 개별 파일로도 저장
            doc_path = output_dir / f"{disc.rcept_no}.json"
            with open(doc_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)

        # 전체 목록 저장
        index_path = output_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({
                "total": len(documents),
                "documents": [
                    {
                        "id": d["id"],
                        "corp_name": d["metadata"]["corp_name"],
                        "report_name": d["metadata"]["report_name"],
                        "report_date": d["metadata"]["report_date"],
                    }
                    for d in documents
                ]
            }, f, ensure_ascii=False, indent=2)

        print(f"\nRAG 문서 내보내기 완료: {output_dir}")
        print(f"총 {len(documents)}건")

        return documents


def main():
    """메인 실행"""
    import argparse

    parser = argparse.ArgumentParser(description="DART 공시 데이터 수집")
    parser.add_argument(
        "--mode",
        choices=["major", "recent", "all"],
        default="major",
        help="수집 모드: major(주요기업), recent(최근공시), all(전체)"
    )
    parser.add_argument(
        "--start-date",
        help="시작일 (YYYYMMDD)"
    )
    parser.add_argument(
        "--end-date",
        help="종료일 (YYYYMMDD)"
    )
    parser.add_argument(
        "--max-count",
        type=int,
        default=500,
        help="최대 수집 건수"
    )
    parser.add_argument(
        "--no-content",
        action="store_true",
        help="원문 다운로드 생략"
    )

    args = parser.parse_args()

    collector = DARTCollector()

    if args.mode == "major":
        print("=== 주요 기업 공시 수집 ===")
        disclosures = collector.collect_major_corps(
            start_date=args.start_date,
            end_date=args.end_date,
            download_content=not args.no_content,
            max_per_corp=args.max_count // len(collector.MAJOR_CORPS),
        )
    elif args.mode == "recent":
        print("=== 최근 공시 수집 ===")
        disclosures = collector.collect_by_report_type(
            report_type="A",  # 정기공시
            start_date=args.start_date,
            end_date=args.end_date,
            download_content=not args.no_content,
            max_count=args.max_count,
        )
    else:
        print("=== 전체 공시 수집 ===")
        # 여러 유형 수집
        disclosures = []
        for report_type in ["A", "B", "I"]:  # 정기, 주요사항, 거래소
            disclosures.extend(
                collector.collect_by_report_type(
                    report_type=report_type,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    download_content=not args.no_content,
                    max_count=args.max_count // 3,
                )
            )

    # 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collector.save_disclosures(disclosures, f"disclosures_{args.mode}_{timestamp}.json")

    # RAG용 내보내기
    collector.export_for_rag(disclosures)

    print("\n=== 수집 완료 ===")


if __name__ == "__main__":
    main()
