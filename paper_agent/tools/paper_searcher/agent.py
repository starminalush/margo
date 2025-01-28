import json
from typing import (
    Annotated,
    Sequence,
    TypedDict,
)

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from paper_agent.tools.paper_searcher.arxiv_tool import arxiv_tool


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    found_results: list[dict]


llm = ChatOpenAI(model="gpt-4o-mini")


tools = [arxiv_tool]

model = llm.bind_tools(tools)


tools_by_name = {tool.name: tool for tool in tools}


# Define our tool node
def tool_node(state: AgentState):
    found_papers = []
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(str(tool_result)),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
        found_papers.extend([t.dict() for t in tool_result])
    return {"messages": outputs, "found_results": found_papers}


# Define the node that calls the model
def call_model(
    state: AgentState,
    config: RunnableConfig,
):
    system_prompt = SystemMessage(
        "Ты умеешь только искать статьи, которые подходят под запрос пользователя. Не отвечай на вопрос, просто найди статью."
    )
    response = model.invoke([system_prompt] + state["messages"], config)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as agent
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        # If tools, then we call the tool node.
        "continue": "tools",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from tools to agent.
# This means that after tools is called, agent node is called next.
workflow.add_edge("tools", "agent")

# Now we can compile and visualize our graph
search_agent = workflow.compile()


def search_papers(question: str):
    answer = search_agent.invoke({"messages": [question]})
    return answer["found_results"]
