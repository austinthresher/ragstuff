import os
import sys
import asyncio
from rich import print as pp

from langchain_core.tools.simple import Tool
from langchain_openai import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from ragstuff.rag import RAGBackend

async def main():
    llm = ChatOpenAI(
        model_name=os.getenv("RAG_MODEL"),
        api_key=os.getenv("RAG_API_KEY"),
        base_url=os.getenv("RAG_BASE_URL"),
    )

    rag = RAGBackend()

    await rag.add_sources(["super-metroid-speed-faq.md"])

    memory = MemorySaver()
    config = {"configurable": {"thread_id": "000"}}
    agent = rag.create_agent(llm, checkpointer=memory)
    result = await agent.ainvoke(
        {"messages": [HumanMessage("How do I perform a Mockball?")]},
        config,
    )
    result["messages"][-1].pretty_print()
    result["messages"].append(HumanMessage("Thanks!"))
    result = await agent.ainvoke(result, config)
    result["messages"][-1].pretty_print()
    # Uncomment to see all tool calls / etc
    # pp(result)


if __name__ == "__main__":
    asyncio.run(main())
