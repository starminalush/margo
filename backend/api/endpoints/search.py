from fastapi import APIRouter
from pydantic import BaseModel

from paper_agent.agent import get_answer

router = APIRouter()


class PaperSearchAnswer(BaseModel):
    title: str
    content: str
    url: str


class PaperInput(BaseModel):
    question: str


class PaperSearchAnswers(BaseModel):
    search_result: list[PaperSearchAnswer]


@router.post("")
async def get_relevant_answer(question: PaperInput):
    answer = get_answer(question.question)
    return {"content": answer}
