from typing import Annotated, Literal, Sequence

from langchain.globals import set_debug, set_verbose
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.types import Command
from typing_extensions import TypedDict

from paper_agent.tools.rag_tool import rag_app
from paper_agent.tools.summary_tool import summary_app_tool

set_debug(True)
set_verbose(True)

memory = MemorySaver()

members = ["summarizer", "consultant"]
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. "
    " Description of Workers: "
    " - summarizer - Used to provide a brief summary of the paper. Trigger only when the user explicitly asks for an overview, general content, or the main ideas of the paper (e.g., Summarize the paper,  Give a brief overview) "
    " - consultant - Used for answering any specific questions about the paper that require detailed information, clarification, analysis, or references to specific sections (e.g., What pre-training techniques are used in BERT? or Explain how the model achieves bidirectionality) "
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
    paper_url: str
    answer: str


def supervisor_node(
    state: PaperAssistantState,
) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto)


def rag_node(state: PaperAssistantState):
    question = state["messages"][0].content
    loader = PyMuPDFLoader(state["paper_url"])
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)
    r = rag_app.invoke({"question": question, "data": split_docs})
    return Command(
        update={
            "messages": [AIMessage(content=r["generation"], name="consultant")],
            "answer": r["generation"],
        },
        goto="supervisor",
    )


# def research_node(state: PaperAssistantState):
#     question = state["messages"][0].content
#     r = search_agent.invoke({"messages": [question]})
#     logging.getLogger().info(r)
#     return Command(
#         update={
#             "messages": [
#                 AIMessage(content=r["messages"][-1].content, name="paper_searcher")
#             ],
#             "found_results": r["found_results"][0],
#             "answer": r["found_results"]
#         },
#         goto="supervisor",
#     )
#


def summary_node(state: PaperAssistantState):
    loader = PyMuPDFLoader(state["paper_url"])
    docs = loader.load()
    from langchain_text_splitters import CharacterTextSplitter

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)

    result = summary_app_tool.invoke({"contents": [doc.page_content for doc in split_docs]})
    return Command(
        update={
            "answer": result["final_summary"],
            "messages": [AIMessage(content=result["final_summary"], name="summarizer")],
        },
        goto="supervisor",
    )


builder = StateGraph(PaperAssistantState)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("consultant", rag_node)
builder.add_node("summarizer", summary_node)
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}


def get_answer(question: str, paper_url: str = None) -> str:
    result = graph.invoke(
        {"messages": [("user", question)], "paper_url": paper_url},
        config=config,
    )
    return result["answer"]
