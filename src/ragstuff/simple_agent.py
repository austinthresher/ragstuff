import asyncio
from pathlib import Path
import os
from copy import deepcopy
from langchain_core.documents import Document
from langgraph.config import get_stream_writer
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, filter_messages
from langchain.tools import tool
from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition, InjectedState, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from typing import Optional
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
import gnureadline as readline
from rich.pretty import pprint
from textwrap import shorten
from uuid import uuid4

import ragstuff.websearch as websearch
import ragstuff.scrape as scrape

from langchain.chat_models import init_chat_model

# Structured output from the model
class ExtractedURL(BaseModel):
    url: str
    description: str = Field(
        description="A few words explaining why the URL looked interesting or relevant"
    )

class ExtractedURLs(BaseModel):
    urls: list[ExtractedURL] = Field(
        default_factory=list,
        description="A list of URLs that might lead to additional information",
    )

# We'll fill in the URL ourselves and stuff them back into a list
class ExtractedURLFrom(ExtractedURL):
    source: str


class PageSummary(BaseModel):
    extracted_information: list[str] = Field(
        description=("All new information from the page that is relevant to the research topic. Leave blank if none.")
    )
    should_follow_urls: bool = Field(
        description=(
            "True if the page contained any URLs that could lead to additional information."
        )
    )



##
"""
Any time we interact with a URL, we only actually include it if we haven't already seen it.

with how bad some of these search queries are, we're going to have to store them and ask for improvement
when shitty results are all we get

The plan (assuming we've already decided to search):
 - Initialize our running context window to the user query. This will be included for all of the below
 - Given the running context window, ask an LLM to write a search query.
 - Run the search, fetch the URLs, stuff them into a list.
 - Ask a model for the most promising URL and pop that URL from the list
 - Visit the page, extract a summary.
 - If the summary isn't empty, add it to running context, then extract URLs from the page with running context visible to LLM
 - Add any gathered URLs to the list
 - Now we can repeat until we are out of URLs or hit some condition based on filling up our running context




"""
##


llm = init_chat_model(
    "openai/gpt-4.1-mini",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

class ResearchAgentState(AgentState):
    research_topic: str
    seen_urls: set[str]
    potential_urls: list[ExtractedURLFrom]
    gathered_context: dict[str, PageSummary]
    next_search: str
    next_url: str

    @classmethod
    def create(cls, research_topic: str):
        return cls(
            research_topic=research_topic,
            seen_urls=set(),
            potential_urls=[],
            gathered_context={},
            next_search="",
            next_url="",
        )

    @staticmethod
    def context(state: dict) -> str:
        summaries = "\n\n".join(
            f'<info source="{u}">\n' + "\n".join(s.extracted_information) + "\n</info>"
            for u, s in state["gathered_context"].items()
        )
        return (
            f"<topic>\n{state['research_topic']}\n</topic>\n"
            f"<gathered_info>\n{summaries}\n</gathered_info>\n"
        )

def research_prompt(
    state: ResearchAgentState,
    task: str,
    suffix: str = "Respond directly with no formatting, quoting, or additional output.",
):
    return [
        SystemMessage(
            "You are a research assistant.\n"
            "The following is the current status of the ongoing research:\n"
            f"{ResearchAgentState.context(state)}\n"
            "In order to assist in the research effort, carefully follow the user's instructions."
        ),
        HumanMessage(f"Task:\n{task}\n\n" f"{suffix}"),
    ]

async def generate_search_query(state: ResearchAgentState):
    result = await llm.ainvoke(
        research_prompt(
            state,
            "Write a single specific search term to fill gaps in the gathered information."
        )
    )
    return {"next_search": result.content}

class URL(BaseModel):
    url: str

async def perform_search(state: ResearchAgentState):
    raw_search_results = await websearch.search_web(
        state["next_search"], num_results=10
    )
    for s in raw_search_results:
        if s.url not in state["seen_urls"]:
            state["seen_urls"].add(s.url)
            state["potential_urls"].append(
                ExtractedURLFrom(
                    source=f"search: {state['next_search']}",
                    url=s.url,
                    description=s.preview,
                )
            )
    pending = "\n\n".join(
        f"# URL: {r.url}\n# Description: {r.description}"
        for r in state["potential_urls"]
    )
    result = await llm.with_structured_output(URL).ainvoke(
        research_prompt(
            state,
            f"Select the URL that will best fill gaps in the collected information:\n\n{pending}",
            suffix="",
        )
    )
    for u in state["potential_urls"]:
        if u.url == result.url:
            state["potential_urls"].remove(u)
            return {"next_url": u.url, "next_search": ""}
    # We didn't get a valid response, just pick one
    print(f"NOT A KNOWN URL: {result.url}", flush=True)
    return {"next_url": state["potential_urls"].pop().url, "next_search": ""}


async def visit_page(state: ResearchAgentState) -> ResearchAgentState:
    url = state["next_url"]
    docs = await scrape.load_urls([url])
    content = docs[0].page_content
    if not content:
        return {"next_url": ""}

    result = await llm.with_structured_output(PageSummary).ainvoke(
        research_prompt(
            state,
            f"Extract new information, if any is present.\n\n--\n\n{content}",
            suffix="",
        )
    )
    if not result.extracted_information:
        return {"next_url": ""}
    state["gathered_context"][url] = result
    pprint(result)
    return state


agent = (
    StateGraph(ResearchAgentState)
    .add_node("generate_search_query", generate_search_query)
    .add_node("perform_search", perform_search)
    .add_node("visit_page", visit_page)
    .add_edge(START, "generate_search_query")
    .add_edge("generate_search_query", "perform_search")
    .add_edge("perform_search", "visit_page")
).compile()

async def main():
    config = {"configurable": {"thread_id": "1"}}
    state = ResearchAgentState.create(
        "Compare examples of intended and unintended sequence breaking in the Metroidvania genre of video games."
    )
    result = await agent.ainvoke(state, config)
    pprint(result)
    print(ResearchAgentState.context(result))

if __name__ == "__main__":
    asyncio.run(main())
