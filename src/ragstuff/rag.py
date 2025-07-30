import io
import uuid
import asyncio
from typing import Callable
from pathlib import Path
from more_itertools import batched

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_chroma import Chroma
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ParentDocumentRetriever, MultiVectorRetriever
from langchain_text_splitters.base import TextSplitter
from langchain_text_splitters.markdown import MarkdownTextSplitter
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore, InMemoryByteStore
from langchain_community.document_compressors import FlashrankRerank
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_unstructured import UnstructuredLoader
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_community.vectorstores.utils import filter_complex_metadata
from langgraph.prebuilt import create_react_agent
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate

from ragstuff.websearch import search_web
from ragstuff.scrape import load_urls, docling_load

DEFAULT_RAG_PROMPT = (
    "You are a researcher who uses semantic search to gather information.\n"
    "Given a topic, question, or sample, use the provided tool to find related information.\n"
    "Always make a variety of distinct queries.\n"
    "Searches return text that is semantically similar to the query you provide. "
    "After your initial queries, review the quality and style of the matched text.\n"
    "Use this information to re-write your queries to specifically match the information you seek.\n"
    "Continue making queries until you have enough information to completely answer the prompt.\n"
    "Do not rely on your own knowledge- all of your information must come directly from the "
    "search results."
)

class RAGBackend:
    def __init__(
        self,
        name: str = "chroma",
        n_results: int = 5,
        persist_directory: bool = False,
        embeddings_model: Embeddings = None,
    ):
        self.n_results = n_results
        self.embedding_model = embeddings_model or HuggingFaceEmbeddings(
            model_kwargs={"device": "cuda"}
        )
        self.db = Chroma(
            name,
            embedding_function=self.embedding_model,
            persist_directory=persist_directory,
        )
        self.db_retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": n_results * 4,
                "fetch_k": n_results * 10,
                "lambda_mult": 0.25,
            },
        )
        self.retriever = ContextualCompressionRetriever(
            base_retriever=self.db_retriever,
            base_compressor=FlashrankRerank(top_n=n_results),
        )
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    def source_exists(self, url: str) -> bool:
        results = self.db.get(limit=1, where={"source": {"$eq": url}})
        return results['ids']

    def _file_loader(self, filename: str):
        # Unstructured seems to be better at local files (pdf/text)
        # but does a poor job with web pages
        print(f"loading {filename}", flush=True)
        return UnstructuredLoader(
            file_path=filename,
            max_characters=1000,
            new_after_n_chars=500,
            chunking_strategy="by_title",
            combine_text_under_n_chars=250,
        )

    def _url_loader(self, url: str):
        # Docling seems to be doing a better job with webpages
        print(f"loading {url}", flush=True)
        return DoclingLoader(file_path=url, export_type=ExportType.DOC_CHUNKS)

    async def search_and_add(self, query: str):
        # Search for more, but keep less. This lets us still find more results
        # if we try searching something similar again.
        search_results = await search_web(query, 10)
        urls = [s.url for s in search_results if not self.source_exists(s.url)]
        print(f"found {len(urls)} new URLs for query: {query}", flush=True)
        if not urls:
            return []
        results = await self.add_urls(urls)
        return results

    async def _load(self, loader):
        try:
            result = await loader.aload()
            return result
        except:
            return []

    async def add_urls(self, urls):
        page_docs = await load_urls(urls)
        chunked_docs = await self.splitter.atransform_documents(page_docs)
        docs = filter_complex_metadata(chunked_docs)
        results = []
        for batch in batched(docs, 1024):
            results += await self.db.aadd_documents(batch)
        return results

    # I'm not sure which works better yet, keeping the docling version for now
    async def add_urls_docling(self, urls):
        chunked_docs = await asyncio.gather(
            *[self._load(self._url_loader(u)) for u in urls]
        )
        flattened = filter_complex_metadata(sum(chunked_docs, []))
        docs = await self.splitter.atransform_documents(flattened)
        results = []
        for batch in batched(docs, 1024):
            results += await self.db.aadd_documents(batch)
        return results

    async def add_files(self, filenames):
        page_docs = await asyncio.gather(
            *[self._load(self._file_loader(f)) for f in filenames]
        )
        flattened = filter_complex_metadata(sum(page_docs, []))
        docs = await self.splitter.atransform_documents(flattened)
        results = []
        for batch in batched(docs, 1024):
            results += await self.db.aadd_documents(batch)
        return results

    async def add_sources(self, sources: list[str]):
        """
        Adds files or URLs that are not already present.
        """
        new_sources = [s for s in sources if not self.source_exists(s)]
        urls = {u for u in new_sources if u.startswith("http")}
        paths = {*new_sources} - urls
        results = await asyncio.gather(self.add_files(paths), self.add_files(urls))
        return results

    async def query(self, search_query: str) -> list[Document]:
        result = await self.retriever.ainvoke(search_query)
        return result

    def get_tool(self, tool_name: str, tool_description: str):
        return create_retriever_tool(
            self.retriever,
            tool_name,
            tool_description,
            document_prompt=PromptTemplate.from_template(
                '<context source="{source}">\n{page_content}\n</context>'
            ),
        )

    def create_agent(
        self,
        llm: BaseChatModel,
        extra_tools: list | None = None,
        prompt: str = DEFAULT_RAG_PROMPT,
        *agent_args,
        **agent_kwargs,
    ):
        retrieve = self.get_tool(
            "search",
            (
                "Search the vectorstore for related text, using a distance-based metric. "
                "Returns text that is semantically similar to the search query."
            ),
        )
        all_tools = [retrieve] + (extra_tools or [])
        return create_react_agent(
            llm,
            tools=all_tools,
            prompt=prompt,
            *agent_args,
            **agent_kwargs,
        )
