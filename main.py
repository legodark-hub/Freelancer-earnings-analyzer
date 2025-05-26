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
    # Read the first 5 rows to get column names and example values
    try:
        df_sample = pd.read_csv(config.CSV_PATH, nrows=5)
    except FileNotFoundError:
        typer.echo(f"Error: The file {config.CSV_PATH} was not found.")
        raise typer.Exit(code=1)
    except pd.errors.EmptyDataError:
        typer.echo(f"Error: The file {config.CSV_PATH} is empty.")
        raise typer.Exit(code=1)

    column_names = df_sample.columns.tolist()
    column_examples_str = "\n".join(
        [
            f"- '{col}': (e.g., {', '.join(df_sample[col].dropna().unique()[:3].astype(str))})"
            for col in column_names
        ]
    )
    system_message_content = f"""You are an assistant that helps answer questions based on data in a CSV file.
The CSV file is located at: {config.CSV_PATH}.
The available columns in this CSV file (with some example values) are:
{column_examples_str}
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
            "A Python REPL. Use this to execute Python code to analyze the CSV file. "
            f"The CSV is at '{config.CSV_PATH}'. Available columns: {', '.join(column_names)}. "
            "Your input to this tool should be ONLY the Python code you want to execute. "
            "For example, to count payment methods, your input might be: "
            f"'import pandas as pd; df = pd.read_csv(\"{config.CSV_PATH}\"); print(df[\"Payment_Method\"].value_counts())'. "
            "Do NOT write 'python_repl(...)' in your input. Just provide the Python code directly. "
            "The code will be executed, and its print output (the result of the last expression or explicit print statements) will be returned. "
            "Ensure your Python code loads the dataframe using pandas and references columns correctly from the list: "
            f"{', '.join(column_names)}."
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
