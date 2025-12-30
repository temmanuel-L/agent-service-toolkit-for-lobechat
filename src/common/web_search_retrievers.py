# -*- coding: utf-8 -*-
"""
@Time ： 2025/10/15 14:38
@Auth ： luanxing
@File ：web_search_retrievers.py
@IDE ：PyCharm
"""

"""
用于封装各类能与langchain整合的网络搜索的retriever
"""

from typing import List, Optional, Dict, Any
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import Field
import warnings

# 抑制包名更改警告
warnings.filterwarnings("ignore", message="This package.*has been renamed.*")

try:
    from ddgs import DDGS  # 新的包名
except ImportError:
    # 如果新包不存在，尝试使用旧包
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        raise ImportError("请安装 ddgs 包: pip install ddgs")


class DuckDuckGoRetriever(BaseRetriever):
    """
    完全修正版 DuckDuckGo Retriever - 适配最新 API
    """

    max_results: int = Field(default=5, description="最大返回结果数量")
    region: str = Field(default="wt-wt", description="搜索区域")
    safesearch: str = Field(default="moderate", description="安全搜索级别")
    timelimit: Optional[str] = Field(default=None, description="时间限制")

    class Config:
        arbitrary_types_allowed = True

    def __init__(
            self,
            max_results: int = 5,
            region: str = "wt-wt",
            safesearch: str = "moderate",
            timelimit: Optional[str] = None,
            **kwargs
    ):
        super().__init__(
            max_results=max_results,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
            **kwargs
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        执行搜索并返回 Document 列表 - 使用正确的 API 参数
        """
        try:
            ddgs = DDGS()

            # 清理查询字符串
            clean_query = self._clean_query(query)

            print(f"执行搜索: {clean_query}")

            # 使用正确的参数名: query 而不是 keywords
            results = ddgs.text(
                query=clean_query,  # 关键修正：使用 query 参数
                max_results=self.max_results,
                region=self.region,
                safesearch=self.safesearch,
                timelimit=self.timelimit,
            )

            # 检查结果是否为空
            if not results:
                return [Document(
                    page_content="未找到相关搜索结果",
                    metadata={"warning": "empty_results", "query": clean_query}
                )]

            return self._format_results(results)

        except Exception as e:
            error_msg = f"搜索错误: {str(e)}"
            print(f"错误详情: {error_msg}")
            return [Document(
                page_content=error_msg,
                metadata={"error": True, "query": query}
            )]

    def _clean_query(self, query: str) -> str:
        """清理查询字符串"""
        return query.strip()

    def _format_results(self, results: List[Dict[str, Any]]) -> List[Document]:
        """格式化搜索结果"""
        documents = []
        for i, result in enumerate(results):
            # 提取关键信息
            title = result.get('title', f'结果 {i + 1}')
            body = result.get('body', '')
            href = result.get('href', '')

            # 构建页面内容
            # page_content = f"标题: {title}\n内容: {body}"
            # if href:
            #     page_content += f"\n链接: {href}"
            page_content = body

            # 构建元数据
            metadata = {
                "source": href,
                "title": title,
                "source_type": "web_search",
                "result_index": i
            }

            document = Document(
                page_content=page_content,
                metadata=metadata
            )
            documents.append(document)

        return documents


# 测试代码
if __name__ == "__main__":
    print("=== 基本搜索测试 ===")

    # 测试基本功能
    retriever = DuckDuckGoRetriever(max_results=2)

    test_queries = [
        "Python programming",
        "machine learning",
        "artificial intelligence"
    ]

    for query in test_queries:
        print(f"\n搜索: {query}")
        results = retriever.invoke(query)
        print(results)
        # if results and "error" not in results[0].metadata:
        #     print(f"找到 {len(results)} 条结果:")
        #     for i, doc in enumerate(results, 1):
        #         print(f"{i}. {doc.page_content[:100]}...")
        # else:
        #     print(f"搜索结果: {results[0].page_content}")

    print("\n=== 测试完成 ===")
