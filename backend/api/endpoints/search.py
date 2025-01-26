from fastapi import APIRouter
from pydantic import BaseModel

from paper_agent.agent import get_answer
from paper_agent.tools.paper_searcher.agent import search_papers

router = APIRouter()


class PaperSearchAnswer(BaseModel):
    title: str
    content: str
    url: str


class PaperInput(BaseModel):
    question: str


class SpecificPaperInput(BaseModel):
    paper_url: str
    question: str


class PaperSearchAnswers(BaseModel):
    search_result: list[PaperSearchAnswer]


@router.post("")
async def get_relevant_answer(question: SpecificPaperInput):
    answer = get_answer(question.question, question.paper_url)
    return {"content": answer}


@router.post("/papers")
async def get_papers(question: PaperInput):
    answer = search_papers(question.question)
    return {"papers": answer}
