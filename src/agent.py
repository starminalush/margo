import logging
from typing import Literal, Annotated, Sequence

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

from langgraph.graph import add_messages
from langgraph.types import Command
from langchain_community.document_loaders import WebBaseLoader

from langgraph.graph import StateGraph, START, END
from src.paper_assistant.tools.paper_searcher.agent import search_agent
from src.paper_assistant.tools.rag_tool import rag_app
from src.paper_assistant.tools.paper_founder import found_paper_url
from src.paper_assistant.tools.summary_tool import summary_app_tool

from langchain.globals import set_debug, set_verbose

set_debug(True)
set_verbose(True)

memory = MemorySaver()

members = ["paper_searcher", "paper_downloader", "summarizer"]
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. "
    " Description of Workers: "
    " - paper_searcher - used to find articles on the internet based on a query. "
    " - paper_downloader - used for load paper in memory by url for summarizer. Use it if content is empty. "
    " - summarizer - used for getting brief summary of paper. "
    " You should never call same worker twice. "
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
    selected_paper: str
    content: list


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
    question = state["messages"][0].content
    r = search_agent.invoke({"messages": [question], })
    logging.getLogger().info(r)
    return Command(
        update={
            "messages": [AIMessage(content=r['messages'][-1].content, name="paper_searcher")],
            'found_results': r['found_results']
        },
        goto="supervisor",
    )

def paper_downloader_node(state: PaperAssistantState):
    paper_name = state.get('selected_paper') or state['found_results'][0]['title']
    paper_url = found_paper_url(paper_name)

    loader = WebBaseLoader(paper_url)
    loader.requests_kwargs = {'verify': False}
    docs = loader.load()
    return Command(
        update={
            "content": docs
        },
        goto='supervisor'
    )

def summary_node(state: PaperAssistantState):
    docs = state['content']
    result = summary_app_tool.invoke({"data": docs})
    return Command(
        update={
            "summary": result['final_summary']
        },
        goto='supervisor'
    )




builder = StateGraph(PaperAssistantState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("paper_searcher", research_node)
builder.add_node('paper_downloader', paper_downloader_node)
builder.add_node("summarizer", summary_node)
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}


def get_answer(question: str):
    result = graph.invoke(
        {"messages": [("user", question)]},
        config=config,
    )
    return result


print(get_answer("О чем статья 'BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding'"))
