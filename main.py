import os
import pandas as pd
import typer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import initialize_agent, Tool
from langchain_experimental.tools import PythonREPLTool
from langchain_core.messages import SystemMessage
import config

# CLI app
app = typer.Typer(
    help="CLI for querying freelancer earnings data using LLM and Python tooling"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def init_vectorstore():
    """
    Загружает CSV, разбивает тексты, создаёт эмбеддинги и сохраняет ChromaDB
    """
    df = pd.read_csv(config.CSV_PATH)

    # Загрузка документов из DataFrame
    loader = DataFrameLoader(
        df,
        page_content_column="Job_Category",
    )
    docs = loader.load()

    # Разбивка на чанки
    splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    split_docs = splitter.split_documents(docs)

    # Эмбеддинги
    embeddings = HuggingFaceEmbeddings(model_name=config.MODEL_NAME)

    # Создание и сохранение векторного хранилища
    vectordb = Chroma.from_documents(
        documents=split_docs, embedding=embeddings, persist_directory=config.DB_DIR
    )
    typer.echo("Initialized vectorstore")


@app.command()
def init():
    """
    Инициализирует ChromaDB из CSV-данных
    """
    if not os.path.exists(config.CSV_PATH):
        typer.echo("CSV file not found.")
        raise typer.Exit(code=1)
    init_vectorstore()


@app.command()
def query(
    question: str = typer.Argument(
        ..., help="Natural language question about freelancer earnings"
    ),
):
    """
    Отвечает на вопрос с помощью RetrievalQA и PythonREPLTool
    """
    # Загрузка векторного хранилища
    embeddings = HuggingFaceEmbeddings(model_name=config.MODEL_NAME)
    vectordb = Chroma(persist_directory=config.DB_DIR, embedding_function=embeddings)
    column_names = pd.read_csv(config.CSV_PATH, nrows=1).columns.tolist()
    column_description = (
        f"Доступные колонки в {config.CSV_PATH}: {', '.join(column_names)}."
    )
    system_message_content = f"""You are an assistant that helps answer questions based on data in a CSV file.
The CSV file is located at: {config.CSV_PATH}.
The available columns in this CSV file are: {", ".join(column_names)}.
When generating Python code to analyze this data:
1. You MUST use pandas to load and manipulate the data from '{config.CSV_PATH}'.
2. You MUST ONLY use the column names from the provided list: {", ".join(column_names)}. Do not invent column names. If a column name has spaces or special characters, ensure it is correctly referenced in pandas (e.g., using df['Column Name']).
3. You must match the terms from the question with the available column names before running queries.
4. For calculations involving counts of projects per individual, look for a column that directly represents the total number of projects completed by that individual (e.g., a column named 'Projects_Completed' or similar). Do not assume a 'project_id' column represents individual projects if the CSV is structured with one row per freelancer.
Your goal is to write Python code that directly answers the user's question using the data from '{config.CSV_PATH}' and the available columns.
"""

    def clean_python_code(code: str) -> str:
        """
        Очищает код от лишних символов и тегов, когда LLM выдает код в 
        формате markdown.
        """
        return (
            code.replace("```python", "")
            .replace("```py", "")
            .replace("```", "")
            .strip()
        )

    # Настройка LLM (локальный совместимый endpoint)
    llm = ChatOpenAI(
        model_name="",
        openai_api_key=config.LLM_API_KEY,
        openai_api_base=config.LLM_BASE_URL,
        temperature=0,
    )

    # Инструмент для агрегаций через Python
    python_tool = Tool(
        name="python_repl",
        func=lambda code: PythonREPLTool().run(clean_python_code(code)),
        description=(
            f"Use to perform aggregations and calculations on the CSV file located at '{config.CSV_PATH}' using pandas. "
            f"Ensure your Python code ONLY uses the available column names: {', '.join(column_names)}. "
            f"Load the dataframe directly from '{config.CSV_PATH}'. "
            "The input to this tool must be raw Python code, not a markdown code block."
        ),
    )

    # Инструмент для RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectordb.as_retriever()
    )
    retrieval_tool = Tool(
        name="retrieval",
        func=lambda q: qa_chain.invoke({"query": q})["result"],
        description="Используйте для ответа на вопросы на основе текстовых данных и метаданных",
    )

    # Инициализация агента с двумя инструментами
    agent = initialize_agent(
        tools=[python_tool, retrieval_tool],
        llm=llm,
        agent="zero-shot-react-description",
        agent_kwargs={"system_message": SystemMessage(content=system_message_content)},
        verbose=True,
    )

    # Запуск агента
    response = agent.invoke({"input": question})
    typer.echo(response["output"])


if __name__ == "__main__":
    app()

# Примеры запросов:
# "Какой процент экспертов выполнил менее 100 проектов?"
# "Насколько выше доход у фрилансеров, принимающих оплату в криптовалюте?"
# "Покажи распределение доходов по регионам клиентов."
# "Какие категории работ имеют наибольшую среднюю почасовую ставку?"
# "Есть ли корреляция между расходами на маркетинг и доходами?"
