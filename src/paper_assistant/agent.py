import logging
from typing import Literal, Annotated, Sequence

from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from langgraph.graph import add_messages
from langgraph.types import Command


from langgraph.graph import StateGraph, START, END
from paper_assistant.tools.paper_searcher.agent import search_agent
from paper_assistant.tools.rag_tool import rag_app

from langchain.globals import set_debug, set_verbose

set_debug(True)
set_verbose(True)

memory = MemorySaver()

members = ["paper_searcher"]
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. "
    " Description of Workers: "
    " - paper_searcher - used to find articles on the internet based on a query. "
    "When finished,"
    " respond with FINISH."
)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]


llm = ChatOpenAI(model="gpt-4o-mini")


class PaperAssistantState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    found_results: list[dict]


def supervisor_node(state: PaperAssistantState) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto)


def rag_node(state: PaperAssistantState):
    question = state["messages"][-1]
    r = rag_app.invoke(input={"question": question, "found_results": state["found_results"]})
    return Command(
        update={
            "messages": [HumanMessage(content=r["messages"][-1].content, name="consultant")],
            "found_results": r["found_results"],
        },
        goto="supervisor",
    )


def research_node(state: PaperAssistantState):
    question = state["messages"][0]
    r = search_agent.invoke({"input": question, "chat_history": []})
    logging.getLogger().info(r)
    return Command(
        update={
            "messages": [HumanMessage(content=r['main_body'], name="paper_searcher")],
        },
        goto="supervisor",
    )


builder = StateGraph(PaperAssistantState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("paper_searcher", research_node)
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}


def get_answer(question: str):
    result = graph.invoke(
        {"messages": [("user", question)]},
        config=config,
    )
    return result
