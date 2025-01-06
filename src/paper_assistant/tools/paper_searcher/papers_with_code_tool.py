from typing import Any, Type

import httpx
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class _PapersWithCodeInput(BaseModel):
    query: str


class _PapersWithCodeOutput(BaseModel):
    paper_url: str
    content: str
    title: str


class PapersWithCodeSearchTool(BaseTool):
    """Tool that find papers on PaperWithCode."""

    name: str = "papers_with_code_search"
    description: str = (
        "A wrapper around PapersWithCode. "
        "Useful for when you need to find papers on PapersWithCode. "
        "Input should be a search query."
    )
    api_base_url: str = "https://paperswithcode.com/api/v1"
    args_schema: Type[BaseModel] = _PapersWithCodeInput

    def _run(self, query: str, num_samples: int = 10, run_manager: CallbackManagerForToolRun | None = None) -> Any:
        search_api_url: str = f"{self.api_base_url}/search/"
        paper_base_url: str = "https://paperswithcode.com/paper"
        params = {"q": query, "items_per_page": num_samples}
        response = httpx.get(search_api_url, params=params).json()

        return [
            _PapersWithCodeOutput(
                paper_url=f"{paper_base_url}/{r['paper']['id']}",
                content=r["paper"]["abstract"],
                title=r["paper"]["title"],
            )
            for r in response["results"]
        ]


papers_with_code_tool = PapersWithCodeSearchTool()
