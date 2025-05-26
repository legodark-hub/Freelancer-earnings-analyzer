import pandas as pd
import typer
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain_experimental.tools import PythonREPLTool
from langchain_core.messages import SystemMessage
import config

app = typer.Typer(
    help="CLI for querying freelancer earnings data using LLM and Python tooling"
)


def clean_python_code(code: str) -> str:
    """
    Очищает код от лишних символов и тегов, когда LLM выдает код в
    формате markdown.
    """
    return code.replace("```python", "").replace("```py", "").replace("```", "").strip()


@app.command()
def query(
    question: str = typer.Argument(
        ..., help="Запрос про заработки фрилансеров на естесвенном языке"
    ),
):
    """
    Отвечает на вопрос с помощью PythonREPLTool
    """
    column_names = pd.read_csv(config.CSV_PATH, nrows=1).columns.tolist()
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
    llm = ChatOpenAI(
        model_name="",
        openai_api_key=config.LLM_API_KEY,
        openai_api_base=config.LLM_BASE_URL,
        temperature=0,
    )

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

    agent = initialize_agent(
        tools=[python_tool],
        llm=llm,
        agent="zero-shot-react-description",
        agent_kwargs={"system_message": SystemMessage(content=system_message_content)},
        verbose=True,
    )

    response = agent.invoke({"input": question})
    typer.echo(response["output"])


if __name__ == "__main__":
    app()
