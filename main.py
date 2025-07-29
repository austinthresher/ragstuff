import os
import sys
import asyncio
from collections import defaultdict
import dotenv

dotenv.load_dotenv()

from constants import DEFAULT_MODEL, BASE_URL, API_KEY

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    AIMessageChunk,
)
from langchain_unstructured import UnstructuredLoader

from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState
from rich import print as pp

from rag import RAGBackend

def collect_chunks(chunks: list[AIMessageChunk]) -> AIMessage:
    if not chunks:
        return AIMessage("")
    ai_message = chunks[0]
    for m in chunks[1:]:
        ai_message += m
    return ai_message

async def stream(llm, prompt, outfile=None, quiet=False) -> AIMessage:
    chunks = []
    async for chunk in llm.astream(prompt):
        chunks.append(chunk)
        if outfile:
            print(chunk.content, flush=True, end="", file=outfile)
        if not quiet:
            print(chunk.content, flush=True, end="")
    if outfile:
        print("", file=fout)
    if not quiet:
        print()
    return collect_chunks(chunks)


async def main():
    # llm = ChatOpenAI(model_name=DEFAULT_MODEL, api_key=API_KEY, base_url=BASE_URL)

    rag = RAGBackend("knowledge", persist=False)
    await rag.add_urls(["https://python.langchain.com/docs/concepts/vectorstores/"])
    #await rag.search_and_add("nonlinear game design")
    result = await rag.query("embedding")

    from rich import print as pp
    pp(result)

    # retriever_tool = rag.get_tool("search", "search the local knowledgebase about Super Metroid")
    # print(retriever_tool.invoke({"query": "mockball"}))
    # agent = create_react_agent(llm, tools=[retriever_tool])
    # result = await agent.ainvoke(
    #     {"messages": [HumanMessage("How do I perform a Mockball in Super Metroid?")]}
    # )
    # result["messages"][-1].pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
