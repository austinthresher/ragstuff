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


llm = init_chat_model(
    "openai/gpt-4.1-nano",
    #"google/gemini-2.5-pro",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


class ResearchAgentState(AgentState):
    visited_urls: set[str] = Field(default_factory=set)


class ShouldResearchResult(BaseModel):
    reasearch_would_help: bool = Field(
        description="Would performing research before answering the query help give a better answer?"
    )

should_research_pipeline = ChatPromptTemplate.from_messages(
    [
        SystemMessage("Task: Should research be performed before answering?"),
        MessagesPlaceholder("history"),
    ]
) | llm.with_structured_output(ShouldResearchResult)

async def should_research_edge(state: ResearchAgentState) -> str:
    history = filter_messages(state.messages, include_types=[HumanMessage, AIMessage])
    result = should_research_pipeline.ainvoke({"history": history})
    return "research" if result["should_research"] else "chat"




class SearchResults(BaseModel):
    urls_to_follow: list[str] = Field(
        description="URLs that will lead to relevant information",
        default_factory=list
    )



class PageContentsResults(SearchResults):
    subtopic: Optional[str] = Field(
        description="The specific topic of `content_summary`, relative to the current topic.",
        default=""
    )
    content_summary: Optional[str] = Field(
        description="Content relevant to the topic, if any.",
        default=""
    )

@task
async def fetch_url(url: str):
    writer = get_stream_writer()
    writer(f"fetch: {url}")
    try:
        pages = await scrape.load_html([url])
        if pages:
            return pages[0].page_content
    except Exception as e:
        writer(f"exception: {type(e).__name__} {e}")
    return ""

class PageSummary(BaseModel):
    url: str
    summary: str

class PageSummaries(BaseModel):
    url: str
    summaries: list[PageSummary]

@task
async def visit_and_summarize_url(
    url: str, context: str, visited: set, max_depth: int
) -> PageSummaries:
    if url in visited or max_depth < 0:
        return PageSummaries(url=url, summaries=[])
    content = await fetch_url(url)
    visited.add(url)
    if not content:
        return PageSummaries(url=url, summaries=[])
    result = await llm.with_structured_output(PageContentsResults).ainvoke(
        [
            SystemMessage(
                "Task: Extract content that could assist answering the query and collect URLs that could also assist. Leave blank if not useful."
            ),
            HumanMessage(
                f"<query>\n{context}\n</query>\n\n<content>\n{content}\n</content>"
            ),
        ],
        config={"configurable": {"thread_id": str(uuid4())}},
    )
    writer = get_stream_writer()
    writer(
        f"({len(result.urls_to_follow or [])}) {shorten(result.content_summary or '', 80)}"
    )
    pages = []
    if result.content_summary:
        pages.append(PageSummary(url=url, summary=result.content_summary))
    if result.subtopic:
        context = f"{context}\n{result.subtopic}"
    for u in result.urls_to_follow or []:
        summaries = await visit_and_summarize_url(u, context, visited, max_depth - 1)
        pages.extend(summaries.summaries)

    return PageSummaries(url=url, summaries=pages)

@task
async def recursive_web_search(
    query: str, context: str, visited: set, max_depth: int = 3
) -> list[PageSummary]:
    raw_search_results = await websearch.search_web(query, num_results=10)
    search_results = "\n\n".join(
        f"# Title: {r.title}\n# URL: {r.url}\n# Preview: {r.preview}"
        for r in raw_search_results or []
    )
    result = await llm.with_structured_output(SearchResults).ainvoke(
        [
            SystemMessage("Task: Gather URLs that appear relevant"),
            HumanMessage(
                f"<context>\n{context}\n</context>\n\n<search_results>\n{search_results}\n<search_results>"
            ),
        ],
        config={"configurable": {"thread_id": str(uuid4())}},
    )
    summaries = []
    for u in result.urls_to_follow or []:
        page = await visit_and_summarize_url(u, context, visited, max_depth)
        summaries.extend(page.summaries or [])
    return summaries


class ChatSummary(BaseModel):
    current_topic: str = Field(
        description="The topic of conversation or inquiry, used to give individual messages context.",
    )

chat_summary_pipeline = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            "Task: Summarize the topic or subject of the user's most recent query"
        ),
        MessagesPlaceholder("history"),
    ]
) | llm.with_structured_output(ChatSummary)

@task
async def summarize_topic(messages: list) -> str:
    history = filter_messages(messages, include_types=[HumanMessage, AIMessage])
    summary = await chat_summary_pipeline.ainvoke(
        {"history": history}, config={"configurable": {"thread_id": str(uuid4())}}
    )
    return summary.current_topic

@task
async def summarize_sources(query: str, topic: str, sources: list[PageSummary]) -> str:
    content = "\n\n".join(
        f'<summary url="{p.url}">\n{p.summary}\n</summary>' for p in sources
    )
    return await llm.invoke(
        [
            SystemMessage(
                "Task: Condense the information contained in these summaries.\n"
                f"Topic: {topic}\n"
                f"Original Query: {query}\n"
            ),
            HumanMessage(content),
        ]
    )


@task
async def condense_sources(query: str, topic: str, sources: list[PageSummary]) -> str:
    if len(sources) <= 10:
        return await summarize_sources(query, topic, sources)
    a = sources[len(sources)//2:]
    b = sources[:len(sources)//2]
    summary_a = await condense_sources(query, topic, a)
    summary_b = await condense_sources(query, topic, b)
    return f"{summary_a}\n\n{summary_b}"

# FIXME: Somehow I got:
"""
================================= Tool Message =================================
Name: search_web

Error: TypeError("object AIMessage can't be used in 'await' expression")
 Please fix your mistakes.
"""

@tool
async def search_web(query: str, state: Annotated[dict, InjectedState]) -> str:
    """Searches the web for the given query, recursively visits pages, and returns relevant information."""
    context = await summarize_topic(state["messages"])
    pages = await recursive_web_search(query, context, state["visited_urls"])
    if not pages:
        return ""
    if len(pages) <= 10:
        return "\n\n".join(
            f'<summary url="{p.url}">\n{p.summary}\n</summary>' for p in pages
        )
    return await condense_sources(query, context, pages)

llm_with_tools = llm.bind_tools(tools=[search_web])
tool_node = ToolNode(tools=[search_web])

async def chat(state: ResearchAgentState):
    response = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [response]}


agent = (
    StateGraph(ResearchAgentState)
    .add_node("chat", chat)
    .add_node("tools", tool_node)
    .add_conditional_edges("chat", tools_condition)
    .add_edge("tools", "chat")
    .add_edge(START, "chat")
).compile()

# create_react_agent(
#     llm,
#     tools=[search_web],
#     prompt="Task: Answer user queries by using the `search_web` tool.",
#     state_schema=ResearchAgentState,
#     version="v1",
#     checkpointer=InMemorySaver()
# )

async def main():
    histfile = Path.cwd() / ".history"
    if not histfile.exists():
        histfile.write_text("")
    histfile = str(histfile)
    readline.read_history_file(histfile)
    config = {"configurable": {"thread_id": "1"}}
    os.environ["TERM"] = "xterm"

    state = ResearchAgentState()
    state["messages"] = []
    state["visited_urls"] = set()
    while True:
        try:
            user_message = input("> ")
            if user_message.startswith("/quit"):
                break
            state["messages"].append(HumanMessage(user_message))
            async for mode, event in agent.astream(
                state,
                stream_mode=["updates", "custom"],
                config=config,
            ):
                try:
                    for node_name, node in event.items():
                        try:
                            for m in node["messages"]:
                                m.pretty_print()
                        except:
                            print(f" {node_name} ".center(80, "="), flush=True)
                            if node_name == "fetch_url":
                                pprint(shorten(str(node), 1024))
                            else:
                                pprint(node, max_length=1024)
                except:
                    print(f" {mode} ".center(80, "="), flush=True)
                    pprint(event, max_length=256)

        except KeyboardInterrupt:
            print("\nInterrupted.", flush=True)
            continue
        except EOFError:
            print("", flush=True)
            return
        finally:
            readline.write_history_file(histfile)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
