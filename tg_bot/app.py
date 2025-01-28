import asyncio
import logging
from os import getenv

import aiohttp
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message

logging.basicConfig(level=logging.INFO)

# Объект бота
bot = Bot(token=getenv("TG_BOT_TOKEN"))

storage = MemoryStorage()
dp = Dispatcher(storage=storage)

TG_HELLO_TEXT = """
Привет!👋

Я - Марго, твой персональный помощник в ресерче и обучении. Я помогу тебе:

быстрее читать статьи и находить в них нужную информацию
находить для тебя статьи в интернете по свободному запросу
расшифровывать, что нарисовано на схеме архитектуры очередной новейшей модели
искать ответы на вопросы по видео
оптимизировать твои заметки
...и многое другое
Но пока я умею только делать саммари статей и отвечать на вопросы
"""


# Создаем группу состояний
class ArticleStates(StatesGroup):
    choosing_topic = State()
    asking_questions = State()
    select_paper = State()


@dp.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    await state.set_state(ArticleStates.choosing_topic)
    await message.answer(TG_HELLO_TEXT + "\n\nНапиши, какую тему ты хочешь исследовать.")


@dp.message(ArticleStates.choosing_topic)
async def choose_topic(message: types.Message, state: FSMContext):
    async with aiohttp.ClientSession() as session:
        async with session.post(getenv("BACKEND_API_URL") + "/papers", json={"question": message.text}) as resp:
            topics = (await resp.json()).get("papers", [])

    if topics:
        # Сохраняем список статей в данные состояния
        await state.update_data(topics=topics)

        # Формируем текст с выбором
        reply_text = "Я нашла несколько статей по твоему запросу:\n\n"
        for i, topic in enumerate(topics, start=1):
            reply_text += f"{i}. {topic['title']}\n"
        reply_text += "\nВыбери номер статьи, чтобы продолжить."

        await message.answer(reply_text)
        await state.set_state(ArticleStates.select_paper)
    else:
        await message.answer("К сожалению, я ничего не нашла по твоему запросу. Попробуй уточнить запрос.")


# Хэндлер для выбора конкретной статьи из списка
@dp.message(ArticleStates.select_paper, F.text.regexp(r"^\d+$"))
async def choose_article(message: types.Message, state: FSMContext):
    user_data = await state.get_data()
    topics = user_data.get("topics", [])

    article_index = int(message.text) - 1
    if 0 <= article_index < len(topics):
        chosen_article = topics[article_index]
        await state.update_data(chosen_article=chosen_article)

        await state.set_state(ArticleStates.asking_questions)
        await message.answer(
            f"Ты выбрал статью: {chosen_article['title']}\n\nТеперь ты можешь задавать вопросы по этой статье."
        )
    else:
        await message.answer("Пожалуйста, выбери корректный номер статьи из списка.")


@dp.message(ArticleStates.asking_questions)
async def handle_question(message: Message, state: FSMContext):
    user_data = await state.get_data()
    chosen_article = user_data.get("chosen_article")

    if not chosen_article:
        await message.answer("Что-то пошло не так. Попробуй выбрать статью снова.")
        await state.set_state(ArticleStates.choosing_topic)
        return

    async with aiohttp.ClientSession() as session:
        async with session.post(
            getenv("BACKEND_API_URL"), json={"paper_url": chosen_article["paper_url"], "question": message.text}
        ) as resp:
            if resp.status == 200:
                answer_text = (await resp.json()).get("content", "Не удалось получить ответ.")
                await message.answer(answer_text)
            else:
                await message.answer("Произошла ошибка при получении ответа. Попробуй снова.")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
