import operator
from typing import Annotated, List, Literal, TypedDict

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents.reduce import (
    collapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

token_max = 1000
llm = ChatOpenAI(model="gpt-4o-mini")
map_prompt = ChatPromptTemplate.from_messages([("system", "Write a concise summary of the following:\\n\\n{context}")])
reduce_template = """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary
of the main themes. Answer in russian.
"""

reduce_prompt = ChatPromptTemplate([("human", reduce_template)])


def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents."""
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)


class OverallState(TypedDict):
    data: str
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str


class SummaryState(TypedDict):
    content: str


def generate_summary(state: SummaryState):
    from langchain_text_splitters import CharacterTextSplitter

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(state['data'])
    prompt = map_prompt.invoke(state[])
    response = llm.invoke(prompt)
    return {"summaries": [response.content], 'contents': split_docs}


def map_summaries(state: OverallState):
    return [Send("generate_summary", {"content": content}) for content in state["contents"]]


def collect_summaries(state: OverallState):
    return {"collapsed_summaries": [Document(summary) for summary in state["summaries"]]}


def _reduce(input: dict) -> str:
    prompt = reduce_prompt.invoke(input)
    response = llm.invoke(prompt)
    return response.content


def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(state["collapsed_summaries"], length_function, token_max)
    results = []
    for doc_list in doc_lists:
        results.append(collapse_docs(doc_list, _reduce))

    return {"collapsed_summaries": results}


def should_collapse(
    state: OverallState,
) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"


def generate_final_summary(state: OverallState):
    response = _reduce(state["collapsed_summaries"])
    return {"final_summary": response}


graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)  # same as before
graph.add_node("collect_summaries", collect_summaries)
graph.add_node("collapse_summaries", collapse_summaries)
graph.add_node("generate_final_summary", generate_final_summary)

graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("generate_final_summary", END)

summary_app_tool = graph.compile()


paper_url = "https://arxiv.org/abs/1810.04805"
loader = WebBaseLoader(paper_url)
loader.requests_kwargs = {'verify': False}
docs = loader.load()


result = summary_app_tool.invoke({"data": docs})


print(result)