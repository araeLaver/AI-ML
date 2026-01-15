"""
금융 뉴스 수집기

네이버 금융 뉴스를 수집하여 RAG 시스템에 활용합니다.
"""

import os
import re
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, quote

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()


@dataclass
class NewsArticle:
    """뉴스 기사"""
    article_id: str
    title: str
    content: str
    published_at: str
    source: str
    url: str
    category: str
    related_stocks: list[str]


class NaverFinanceNewsCollector:
    """네이버 금융 뉴스 수집기"""

    BASE_URL = "https://finance.naver.com"

    # 주요 종목 코드
    MAJOR_STOCKS = {
        "005930": "삼성전자",
        "000660": "SK하이닉스",
        "373220": "LG에너지솔루션",
        "207940": "삼성바이오로직스",
        "006400": "삼성SDI",
        "005380": "현대차",
        "000270": "기아",
        "068270": "셀트리온",
        "005490": "POSCO홀딩스",
        "105560": "KB금융",
        "055550": "신한지주",
        "035420": "NAVER",
        "035720": "카카오",
        "086790": "하나금융지주",
        "051910": "LG화학",
    }

    # 카테고리
    CATEGORIES = {
        "economy": "경제",
        "finance": "금융",
        "stock": "증권",
        "industry": "산업/재계",
        "global": "글로벌경제",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

        self.data_dir = Path(__file__).parent.parent.parent / "data" / "news"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_soup(self, url: str) -> BeautifulSoup:
        """페이지 파싱"""
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        time.sleep(0.3)  # Rate limiting
        return BeautifulSoup(response.text, "html.parser")

    def collect_stock_news(
        self,
        stock_code: str,
        max_pages: int = 5,
    ) -> list[NewsArticle]:
        """
        종목별 뉴스 수집

        Args:
            stock_code: 종목코드
            max_pages: 최대 페이지 수

        Returns:
            뉴스 목록
        """
        articles = []
        stock_name = self.MAJOR_STOCKS.get(stock_code, stock_code)

        print(f"[{stock_name}] 뉴스 수집 중...")

        for page in range(1, max_pages + 1):
            url = f"{self.BASE_URL}/item/news_news.naver?code={stock_code}&page={page}"

            try:
                soup = self._get_soup(url)
                news_table = soup.select("table.type5 tbody tr")

                for row in news_table:
                    title_elem = row.select_one("td.title a")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    news_url = urljoin(self.BASE_URL, title_elem.get("href", ""))

                    # 날짜 추출
                    date_elem = row.select_one("td.date")
                    pub_date = date_elem.get_text(strip=True) if date_elem else ""

                    # 출처 추출
                    source_elem = row.select_one("td.info")
                    source = source_elem.get_text(strip=True) if source_elem else ""

                    # 기사 ID 추출
                    article_id = self._extract_article_id(news_url)
                    if not article_id:
                        continue

                    # 본문 수집
                    content = self._get_article_content(news_url)
                    if not content or len(content) < 100:
                        continue

                    article = NewsArticle(
                        article_id=article_id,
                        title=title,
                        content=content,
                        published_at=pub_date,
                        source=source,
                        url=news_url,
                        category="stock",
                        related_stocks=[stock_code],
                    )
                    articles.append(article)

                print(f"  - 페이지 {page}: {len(news_table)}건")

            except Exception as e:
                print(f"  - 페이지 {page} 오류: {e}")
                continue

        print(f"  - 총 수집: {len(articles)}건")
        return articles

    def _extract_article_id(self, url: str) -> Optional[str]:
        """URL에서 기사 ID 추출"""
        match = re.search(r"article_id=(\d+)", url)
        if match:
            return match.group(1)
        match = re.search(r"/(\d+)$", url)
        if match:
            return match.group(1)
        return None

    def _get_article_content(self, url: str) -> Optional[str]:
        """기사 본문 수집"""
        try:
            soup = self._get_soup(url)

            # 네이버 금융 뉴스 본문 선택자
            content_elem = soup.select_one("#news_read, .scr01, #content, article")
            if not content_elem:
                return None

            # 텍스트 추출
            text = content_elem.get_text(separator="\n", strip=True)

            # 광고/관련 기사 제거
            text = re.sub(r"\[관련.*?\]", "", text)
            text = re.sub(r"▶.*?(?=\n|$)", "", text)
            text = re.sub(r"기자\s*$", "", text)

            # 공백 정리
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = text.strip()

            return text if len(text) > 100 else None

        except Exception:
            return None

    def collect_category_news(
        self,
        category: str = "economy",
        max_pages: int = 10,
    ) -> list[NewsArticle]:
        """
        카테고리별 뉴스 수집

        Args:
            category: 카테고리 (economy, finance, stock, industry, global)
            max_pages: 최대 페이지 수

        Returns:
            뉴스 목록
        """
        articles = []
        category_name = self.CATEGORIES.get(category, category)

        print(f"[{category_name}] 뉴스 수집 중...")

        # 네이버 뉴스 경제 섹션
        base_url = f"https://news.naver.com/section/101"

        for page in range(1, max_pages + 1):
            url = f"{base_url}?page={page}"

            try:
                soup = self._get_soup(url)
                news_items = soup.select("li.sa_item")

                for item in news_items:
                    title_elem = item.select_one("a.sa_text_title")
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    news_url = title_elem.get("href", "")

                    article_id = self._extract_article_id(news_url)
                    if not article_id:
                        continue

                    # 본문 수집
                    content = self._get_naver_news_content(news_url)
                    if not content:
                        continue

                    # 관련 종목 추출
                    related_stocks = self._extract_related_stocks(title + " " + content)

                    article = NewsArticle(
                        article_id=article_id,
                        title=title,
                        content=content,
                        published_at=datetime.now().strftime("%Y.%m.%d"),
                        source="네이버뉴스",
                        url=news_url,
                        category=category,
                        related_stocks=related_stocks,
                    )
                    articles.append(article)

                print(f"  - 페이지 {page}: {len(news_items)}건")

            except Exception as e:
                print(f"  - 페이지 {page} 오류: {e}")
                continue

        print(f"  - 총 수집: {len(articles)}건")
        return articles

    def _get_naver_news_content(self, url: str) -> Optional[str]:
        """네이버 뉴스 본문 수집"""
        try:
            soup = self._get_soup(url)

            content_elem = soup.select_one("#dic_area, #articeBody, .newsct_article")
            if not content_elem:
                return None

            text = content_elem.get_text(separator="\n", strip=True)
            text = re.sub(r"\n{3,}", "\n\n", text)

            return text if len(text) > 100 else None

        except Exception:
            return None

    def _extract_related_stocks(self, text: str) -> list[str]:
        """텍스트에서 관련 종목 추출"""
        related = []
        for code, name in self.MAJOR_STOCKS.items():
            if name in text:
                related.append(code)
        return related

    def collect_all_major_stocks(
        self,
        max_pages_per_stock: int = 3,
    ) -> list[NewsArticle]:
        """
        주요 종목 전체 뉴스 수집

        Args:
            max_pages_per_stock: 종목당 최대 페이지

        Returns:
            뉴스 목록
        """
        all_articles = []

        for stock_code in self.MAJOR_STOCKS.keys():
            articles = self.collect_stock_news(stock_code, max_pages_per_stock)
            all_articles.extend(articles)

        # 중복 제거
        seen = set()
        unique_articles = []
        for article in all_articles:
            if article.article_id not in seen:
                seen.add(article.article_id)
                unique_articles.append(article)

        return unique_articles

    def save_articles(
        self,
        articles: list[NewsArticle],
        filename: str = "news.json",
    ) -> Path:
        """
        뉴스 저장

        Args:
            articles: 뉴스 목록
            filename: 파일명

        Returns:
            저장 경로
        """
        filepath = self.data_dir / filename

        data = {
            "collected_at": datetime.now().isoformat(),
            "total_count": len(articles),
            "articles": [asdict(a) for a in articles],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n저장 완료: {filepath}")
        print(f"총 {len(articles)}건")

        return filepath

    def export_for_rag(
        self,
        articles: list[NewsArticle],
        output_dir: Optional[Path] = None,
    ) -> list[dict]:
        """
        RAG 시스템용 내보내기

        Args:
            articles: 뉴스 목록
            output_dir: 출력 디렉토리

        Returns:
            RAG용 문서 목록
        """
        if not output_dir:
            output_dir = self.data_dir / "rag_documents"
        output_dir.mkdir(parents=True, exist_ok=True)

        documents = []

        for article in articles:
            # 종목명 추가
            stock_names = [
                self.MAJOR_STOCKS.get(code, code)
                for code in article.related_stocks
            ]

            doc = {
                "id": f"news_{article.article_id}",
                "content": f"{article.title}\n\n{article.content}",
                "metadata": {
                    "source": f"NEWS_{article.article_id}",
                    "title": article.title,
                    "published_at": article.published_at,
                    "news_source": article.source,
                    "url": article.url,
                    "category": article.category,
                    "related_stocks": article.related_stocks,
                    "related_stock_names": stock_names,
                    "type": "news_article",
                },
            }
            documents.append(doc)

            # 개별 파일 저장
            doc_path = output_dir / f"news_{article.article_id}.json"
            with open(doc_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)

        # 인덱스 저장
        index_path = output_dir / "news_index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump({
                "total": len(documents),
                "documents": [
                    {
                        "id": d["id"],
                        "title": d["metadata"]["title"],
                        "published_at": d["metadata"]["published_at"],
                        "related_stocks": d["metadata"]["related_stock_names"],
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

    parser = argparse.ArgumentParser(description="금융 뉴스 수집")
    parser.add_argument(
        "--mode",
        choices=["stocks", "category", "all"],
        default="stocks",
        help="수집 모드"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=3,
        help="최대 페이지 수"
    )

    args = parser.parse_args()

    collector = NaverFinanceNewsCollector()

    if args.mode == "stocks":
        print("=== 주요 종목 뉴스 수집 ===")
        articles = collector.collect_all_major_stocks(args.max_pages)
    elif args.mode == "category":
        print("=== 경제 뉴스 수집 ===")
        articles = collector.collect_category_news("economy", args.max_pages)
    else:
        print("=== 전체 뉴스 수집 ===")
        articles = collector.collect_all_major_stocks(args.max_pages)
        articles.extend(collector.collect_category_news("economy", args.max_pages))

    # 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    collector.save_articles(articles, f"news_{args.mode}_{timestamp}.json")

    # RAG용 내보내기
    collector.export_for_rag(articles)

    print("\n=== 수집 완료 ===")


if __name__ == "__main__":
    main()
