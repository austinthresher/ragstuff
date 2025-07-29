# This is a generic web search interface to use with the rest of
# langchain / langgraph. I'm splitting this out into its own tiny
# file so that the search backend can be relatively easily replaced.
# The proper way to do this would be to use dependency injection.

from pydantic import BaseModel
from langchain_community.utilities import SearxSearchWrapper

from constants import WEB_SEARCH_URL

_search_backend = SearxSearchWrapper(searx_host=WEB_SEARCH_URL)

class SearchResult(BaseModel):
    url: str
    title: str = ""
    preview: str = ""

async def search_web(query: str, num_results: int, **kwargs) -> list[SearchResult]:
    """
    Asynchronously searches the web for the given query and returns a list
    of results.
    """
    search_results = await _search_backend.aresults(
        query, num_results=num_results, **kwargs
    )
    return [
        SearchResult(
            url=r["link"],
            title=r.get("title") or "",
            preview=r.get("snippet") or "",
        )
        for r in search_results
    ]
