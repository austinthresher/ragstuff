import io
import uuid
import asyncio
from typing import Callable
from pathlib import Path
import itertools

from langchain_core.documents import Document
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

import constants
from websearch import search_web
from scrape import load_urls, docling_load


class RAGBackend:
    def __init__(
        self,
        name: str,
        initial_results: int = 10,
        top_n: int = 5,
        persist: bool = False,
    ):
        # Only needed if I bring back ParentDocumentRetriever
        # self.store = (
        #     LocalFileStore(root_path=f"{name}.store")
        #     if persist
        #     else InMemoryByteStore()
        # )

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-mpnet-base-v2", model_kwargs={"device": "cuda"}
        )
        self.db = Chroma(
            f"chroma_{name}",
            embedding_function=self.embedding_model,
            persist_directory=f"{name}.db" if persist else None,
        )
        self.retriever = ContextualCompressionRetriever(
            base_retriever=self.db.as_retriever(search_kwargs={"k": initial_results}),
            base_compressor=FlashrankRerank(
                top_n=top_n, model="ms-marco-MiniLM-L-12-v2"
            ),
        )

    def source_exists(self, url: str) -> bool:
        results = self.db.get(limit=1, where={"source": {"$eq": url}})
        return results['ids']

    def _file_loader(self, filename: str):
        # Unstructured seems to be better at local files (pdf/text)
        # but does a poor job with web pages
        print(f"loading {filename}", flush=True)
        return UnstructuredLoader(
            file_path=filename,
            max_characters=2000,
            new_after_n_chars=1000,
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
        splitter = RecursiveCharacterTextSplitter()
        chunked_docs = await splitter.atransform_documents(page_docs)
        flattened = filter_complex_metadata(chunked_docs)
        results = []
        for batch in itertools.batched(flattened, 1024):
            results += await self.db.aadd_documents(batch)
        return results

    # docling version
    # async def add_urls(self, urls):
    #     chunked_docs = await asyncio.gather(
    #         *[self._load(self._url_loader(u)) for u in urls]
    #     )
    #     flattened = filter_complex_metadata(sum(chunked_docs, []))
    #     results = []
    #     for batch in itertools.batched(flattened, 1024):
    #         results += await self.db.aadd_documents(batch)
    #     return results

    async def add_files(self, filenames):
        chunked_docs = await asyncio.gather(
            *[self._load(self._file_loader(f)) for f in filenames]
        )
        flattened = filter_complex_metadata(sum(chunked_docs, []))
        results = []
        for batch in itertools.batched(flattened, 1024):
            results += await self.db.aadd_documents(batch)
        return results

    async def query(self, search_query: str) -> list[Document]:
        result = await self.retriever.ainvoke(search_query)
        return result

    def get_tool(self, tool_name, tool_description):
        return create_retriever_tool(self.retriever, tool_name, tool_description)
