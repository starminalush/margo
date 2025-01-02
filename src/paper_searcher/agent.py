from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.paper_searcher.arxiv_tool import arxiv_tool
from src.paper_searcher.papers_with_code_tool import papers_with_code_tool
from src.paper_searcher.web_search_tool import web_search_tool

llm = ChatOpenAI(model="gpt-4o-mini")

search_agent = create_react_agent(llm, tools=[web_search_tool, arxiv_tool, papers_with_code_tool])

r = search_agent.invoke(input={"messages": "найти мне статьи по теме ллм агентов"})
