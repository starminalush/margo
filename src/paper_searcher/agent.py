from langchain_openai import ChatOpenAI

from src.paper_searcher.arxiv_tool import arxiv_tool
from src.paper_searcher.papers_with_code_tool import papers_with_code_tool
from src.paper_searcher.web_search_tool import web_search_tool

from langgraph.graph import StateGraph, END

import json
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig


from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

llm = ChatOpenAI(model="gpt-4o-mini")


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    urls: list[str]


tools = [arxiv_tool, papers_with_code_tool, web_search_tool]


model = llm.bind_tools(tools)


tools_by_name = {tool.name: tool for tool in tools}


# Define our tool node
def tool_node(state: AgentState):
    outputs = []
    urls = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(str(tool_result)),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
        urls.extend([r.paper_url for r in tool_result])
    return {"messages": outputs, "urls": urls}


# Define the node that calls the model
def call_model(
    state: AgentState,
    config: RunnableConfig,
):
    system_prompt = SystemMessage(
        "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
    )
    response = model.invoke([system_prompt] + state["messages"], config)
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        # If `tools`, then we call the tool node.
        "continue": "tools",
        "end": END,
    },
)

workflow.add_edge("tools", "agent")

search_agent = workflow.compile()
