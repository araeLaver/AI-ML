"""
수집된 데이터를 RAG 시스템에 로드

DART 공시, 뉴스 데이터를 VectorStore에 인덱싱합니다.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.vectorstore import VectorStoreService
from src.rag.chunking import ChunkerFactory


class RAGDataLoader:
    """RAG 데이터 로더"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        chunking_strategy: str = "recursive",
    ):
        """
        Args:
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
            chunking_strategy: 청킹 전략 (fixed, sentence, recursive, semantic)
        """
        self.vectorstore = VectorStoreService()
        self.chunker = ChunkerFactory.create(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "sources": {},
        }

    def load_dart_disclosures(
        self,
        json_path: Optional[Path] = None,
    ) -> int:
        """
        DART 공시 데이터 로드

        Args:
            json_path: JSON 파일 경로 (없으면 최신 파일 사용)

        Returns:
            로드된 문서 수
        """
        if not json_path:
            # 최신 파일 찾기
            dart_dir = self.data_dir / "dart"
            json_files = list(dart_dir.glob("disclosures_*.json"))
            if not json_files:
                print("DART 공시 데이터 없음")
                return 0
            json_path = max(json_files, key=lambda p: p.stat().st_mtime)

        print(f"\n=== DART 공시 로드: {json_path.name} ===")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        disclosures = data.get("disclosures", [])
        loaded = 0

        for disc in disclosures:
            content = disc.get("content")
            if not content:
                continue

            # 청킹
            chunks = self.chunker.split_text(content)

            for i, chunk in enumerate(chunks):
                doc_id = f"dart_{disc['rcept_no']}_{i}"
                metadata = {
                    "source": f"DART_{disc['rcept_no']}",
                    "corp_name": disc.get("corp_name", ""),
                    "report_name": disc.get("report_nm", ""),
                    "report_date": disc.get("rcept_dt", ""),
                    "type": "dart_disclosure",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }

                self.vectorstore.add_document(
                    doc_id=doc_id,
                    content=chunk,
                    metadata=metadata,
                )

            loaded += 1
            self.stats["total_chunks"] += len(chunks)

            if loaded % 10 == 0:
                print(f"  - {loaded}/{len(disclosures)} 로드됨")

        self.stats["total_documents"] += loaded
        self.stats["sources"]["dart"] = loaded
        print(f"  - DART 완료: {loaded}건, {self.stats['total_chunks']}청크")

        return loaded

    def load_news_articles(
        self,
        json_path: Optional[Path] = None,
    ) -> int:
        """
        뉴스 기사 로드

        Args:
            json_path: JSON 파일 경로

        Returns:
            로드된 문서 수
        """
        if not json_path:
            news_dir = self.data_dir / "news"
            json_files = list(news_dir.glob("news_*.json"))
            if not json_files:
                print("뉴스 데이터 없음")
                return 0
            json_path = max(json_files, key=lambda p: p.stat().st_mtime)

        print(f"\n=== 뉴스 로드: {json_path.name} ===")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        articles = data.get("articles", [])
        loaded = 0
        chunk_count = 0

        for article in articles:
            content = f"{article['title']}\n\n{article['content']}"

            # 청킹
            chunks = self.chunker.split_text(content)

            for i, chunk in enumerate(chunks):
                doc_id = f"news_{article['article_id']}_{i}"
                metadata = {
                    "source": f"NEWS_{article['article_id']}",
                    "title": article.get("title", ""),
                    "published_at": article.get("published_at", ""),
                    "news_source": article.get("source", ""),
                    "category": article.get("category", ""),
                    "type": "news_article",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }

                self.vectorstore.add_document(
                    doc_id=doc_id,
                    content=chunk,
                    metadata=metadata,
                )

            loaded += 1
            chunk_count += len(chunks)

            if loaded % 20 == 0:
                print(f"  - {loaded}/{len(articles)} 로드됨")

        self.stats["total_documents"] += loaded
        self.stats["total_chunks"] += chunk_count
        self.stats["sources"]["news"] = loaded
        print(f"  - 뉴스 완료: {loaded}건, {chunk_count}청크")

        return loaded

    def load_custom_documents(
        self,
        documents_dir: Path,
    ) -> int:
        """
        커스텀 문서 로드 (JSON 파일들)

        Args:
            documents_dir: 문서 디렉토리

        Returns:
            로드된 문서 수
        """
        print(f"\n=== 커스텀 문서 로드: {documents_dir} ===")

        json_files = list(documents_dir.glob("*.json"))
        if not json_files:
            print("문서 없음")
            return 0

        loaded = 0
        chunk_count = 0

        for json_file in json_files:
            if json_file.name in ["index.json", "news_index.json"]:
                continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    doc = json.load(f)

                content = doc.get("content", "")
                if not content:
                    continue

                # 청킹
                chunks = self.chunker.split_text(content)

                for i, chunk in enumerate(chunks):
                    doc_id = f"{doc.get('id', json_file.stem)}_{i}"
                    metadata = doc.get("metadata", {})
                    metadata["chunk_index"] = i
                    metadata["total_chunks"] = len(chunks)

                    self.vectorstore.add_document(
                        doc_id=doc_id,
                        content=chunk,
                        metadata=metadata,
                    )

                loaded += 1
                chunk_count += len(chunks)

            except Exception as e:
                print(f"  - {json_file.name} 오류: {e}")
                continue

        self.stats["total_documents"] += loaded
        self.stats["total_chunks"] += chunk_count
        self.stats["sources"]["custom"] = loaded
        print(f"  - 커스텀 완료: {loaded}건, {chunk_count}청크")

        return loaded

    def load_all(self) -> dict:
        """
        모든 데이터 로드

        Returns:
            로드 통계
        """
        print("\n" + "=" * 50)
        print("RAG 데이터 로드 시작")
        print("=" * 50)

        # DART 공시
        self.load_dart_disclosures()

        # 뉴스
        self.load_news_articles()

        # 커스텀 문서 (있으면)
        custom_dirs = [
            self.data_dir / "dart" / "rag_documents",
            self.data_dir / "news" / "rag_documents",
        ]
        for custom_dir in custom_dirs:
            if custom_dir.exists():
                self.load_custom_documents(custom_dir)

        # VectorStore 저장
        self.vectorstore.persist()

        print("\n" + "=" * 50)
        print("로드 완료!")
        print(f"  - 총 문서: {self.stats['total_documents']}건")
        print(f"  - 총 청크: {self.stats['total_chunks']}개")
        print(f"  - 소스별: {self.stats['sources']}")
        print("=" * 50)

        return self.stats

    def get_stats(self) -> dict:
        """현재 VectorStore 통계"""
        return {
            "loaded": self.stats,
            "vectorstore": self.vectorstore.get_stats(),
        }


def main():
    """메인 실행"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG 데이터 로드")
    parser.add_argument(
        "--source",
        choices=["dart", "news", "all"],
        default="all",
        help="로드할 소스"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="청크 크기"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="청크 오버랩"
    )
    parser.add_argument(
        "--strategy",
        choices=["fixed", "sentence", "recursive", "semantic"],
        default="recursive",
        help="청킹 전략"
    )

    args = parser.parse_args()

    loader = RAGDataLoader(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=args.strategy,
    )

    if args.source == "dart":
        loader.load_dart_disclosures()
    elif args.source == "news":
        loader.load_news_articles()
    else:
        loader.load_all()

    print("\n최종 통계:")
    print(json.dumps(loader.get_stats(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
