from typing import Any, Type

import arxiv
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class _ArxivInput(BaseModel):
    query: str


class _ArxivOutput(BaseModel):
    paper_url: str
    content: str
    title: str
    paper_id: str


class ArxivTool(BaseTool):
    """Tool that find papers on PaperWithCode."""

    name: str = "arxiv_search"
    description: str = (
        "A wrapper around Arxiv. "
        "Useful for when you need to find papers on Arxiv. "
        "Input should be a search query."
    )
    args_schema: Type[BaseModel] = _ArxivInput
    client: arxiv.Client = arxiv.Client()

    def _run(self, query: str, num_samples: int = 10, run_manager: CallbackManagerForToolRun | None = None) -> Any:
        search = arxiv.Search(query=query, max_results=num_samples, sort_by=arxiv.SortCriterion.Relevance)
        results = self.client.results(search)
        return [
            _ArxivOutput(
                paper_url=r.pdf_url,
                content=r.summary,
                title=r.title,
                paper_id=r.entry_id
            )
            for r in results
        ]


arxiv_tool = ArxivTool()
