from typing import Any, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from tavily import TavilyClient


class _WebSearchInput(BaseModel):
    query: str
    num_samples: int


class _WebSearchOutput(BaseModel):
    paper_url: str
    content: str
    title: str


class WebSearchTool(BaseTool):
    """Tool that find papers on web."""

    name: str = "web_search"
    description: str = (
        "A wrapper around TavilySearch. "
        "Useful for when you need to find papers in internet. "
        "Input should be a search query."
    )
    tavily_client: TavilyClient = TavilyClient()
    args_schema: Type[BaseModel] = _WebSearchInput

    def _run(self, query: str, num_samples: int = 10, run_manager: CallbackManagerForToolRun | None = None) -> Any:
        response = self.tavily_client.search(query=query, max_results=num_samples)
        return [
            _WebSearchOutput(
                paper_url=r["url"],
                content=r["content"],
                title=r["title"],
            )
            for r in response["results"]
        ]


web_search_tool = WebSearchTool()
