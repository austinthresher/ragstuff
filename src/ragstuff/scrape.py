from langchain_core.documents import Document
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

from readability import Readability
from html_to_markdown import convert_to_markdown

# Don't return websites that contain these strings because they
# won't have any actual content
BLOCKED_STRINGS = [
    "website is running Anubis",
]

from docling.chunking import HybridChunker
async def docling_load(path_or_url: str) -> Document:
    loader = DoclingLoader(
        file_path=path_or_url,
        chunker=HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2"),
        export_type=ExportType.DOC_CHUNKS,
    )
    result = await loader.aload()
    return result


def cleanup_html_document(
    html_doc: Document, empty_threshold: int = 1000
) -> Document | None:
    metadata = {**html_doc.metadata}
    parser = Readability()
    simple_html, err = parser.parse(html_doc.page_content)
    if err:
        return None
    content = convert_to_markdown(simple_html.content, heading_style="atx", preprocess_html=True)
    if len(content) < empty_threshold:
        return None
    if any(b in content for b in BLOCKED_STRINGS):
        return None
    metadata["title"] = simple_html.title
    metadata["site"] = simple_html.site_name
    metadata["date"] = simple_html.published_time
    return Document(page_content=content, metadata=metadata)


async def load_urls(urls: list[str], playwright: bool = False) -> list[Document]:
    docs = await AsyncHtmlLoader(
        urls,
        ignore_load_errors=True,
        requests_per_second=5,
    ).aload()
    results = []
    for doc in docs:
        if result := cleanup_html_document(doc):
            results.append(result)
    return results
