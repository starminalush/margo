from typing import Literal, Annotated, Sequence

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import add_messages
from langgraph.types import Command

from src.paper_assistant.summary_tool import summary_app_tool

from langgraph.graph import StateGraph, START, END
from src.paper_searcher.agent import search_agent

members = ["paper_searcher", "summarizer"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. "
    " Description of Workers: "
    " - paper_searcher - used to find articles on the internet based on a query. "
    " - summarizer - used to provide a brief summary of a found article or to describe what the article is about when requested. "
    "When finished,"
    " respond with FINISH."
)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]


llm = ChatOpenAI(model="gpt-4o-mini")


class PaperAssistantState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    urls: list[str] | None = None


def supervisor_node(state: PaperAssistantState) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto)


def research_node(state: PaperAssistantState):
    question = state["messages"][0]
    r = search_agent.invoke(input={"messages": [question]})
    return Command(
        update={
            "messages": [HumanMessage(content=r["messages"][-1].content, name="paper_searcher")],
            "urls": r["urls"],
        },
        goto="supervisor",
    )


def summarizer_node(state: PaperAssistantState):
    urls = state["urls"]
    loader = WebBaseLoader(web_path=urls)
    docs = loader.load()

    from langchain_text_splitters import CharacterTextSplitter

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    summary = summary_app_tool.invoke({"contents": [doc.page_content for doc in split_docs]})
    return Command(
        update={"messages": [HumanMessage(content=summary["final_summary"], name="summarizer")], "urls": urls},
        goto="supervisor",
    )


builder = StateGraph(PaperAssistantState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("paper_searcher", research_node)
builder.add_node("summarizer", summarizer_node)
# builder.add_node("paper_consultant", rag_app)
graph = builder.compile()


for s in graph.stream(
    {
        "messages": [
            ("user", "три последние статьи про MoE")
        ]
    },
):
    print(s)
    print("----")
