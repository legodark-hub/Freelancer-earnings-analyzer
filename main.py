import pandas as pd
import typer
import re # For regular expression operations
from langchain_openai import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langgraph.graph import StateGraph, END
from typing import TypedDict, List as TypingList, Optional # type: ignore
import config


app = typer.Typer(
    help="CLI for querying freelancer earnings data using LLM and Python tooling"
)


def clean_python_code(code: str) -> str:
    """
    Извлекает Python код из первого блока ```python ... ``` в строке.
    Убирает всё до "```python" включительно и всё после закрывающего "```" включительно.
    Если блок не найден, возвращает пустую строку.
    """
    # Regex to find content within ```python ... ```
    # It captures the content between "```python" (and any following whitespace)
    # and the next "```" (and any preceding whitespace).
    match = re.search(r"```python\s*([\s\S]*?)\s*```", code, re.DOTALL)
    if match:
        # Group 1 is the content between the markers
        return match.group(1).strip()
    else:
        # If the specific ```python ... ``` block is not found, return an empty string.
        return ""

class GraphState(TypedDict):
    question: str
    csv_path: str
    column_names: TypingList[str]
    column_examples_str: str
    generated_code: Optional[str]
    code_execution_output: Optional[str]
    error_message: Optional[str]


python_repl = PythonREPLTool()


@app.command()
def query(
    question: str = typer.Argument(
        ..., help="Запрос про заработки фрилансеров на естественном языке"
    ),
):
    """
    Отвечает на вопрос с помощью PythonREPLTool
    """
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

    llm = ChatOpenAI(
        model_name=config.MODEL_NAME,
        openai_api_key=config.LLM_API_KEY,
        openai_api_base=config.LLM_BASE_URL,
        temperature=0.5,
    )

    def generate_python_code_node(state: GraphState):
        typer.echo("Generating Python code...")
        prompt = f"""
Your task is to generate a Python script to answer the user's question.
The CSV file is located at: {state["csv_path"]}.
The available columns in this CSV file (with some example values) are:
{state["column_examples_str"]}

User's question: {state["question"]}

Please generate a Python script that:
1. Imports pandas: `import pandas as pd`.
2. Loads the dataframe: `df = pd.read_csv("{state["csv_path"]}")`.
3. Uses ONLY the column names from the provided list: {", ".join(state["column_names"])}. Do not invent column names. If a column name has spaces or special characters, ensure it is correctly referenced in pandas (e.g., using df['Column Name']).
4. Matches the terms from the question with the available column names before running queries.
5. For calculations involving counts of projects per individual, look for a column that directly represents the total number of projects completed by that individual (e.g., a column named 'Projects_Completed' or similar). Do not assume a 'project_id' column represents individual projects if the CSV is structured with one row per freelancer.
6. The script should perform the analysis and print the final result to standard output (e.g., `print(df['some_column'].value_counts())`).
7. Your response MUST NOT include any explanatory text, markdown formatting (like ```python), or anything other than the Python code itself.

Provide only the Python code.
"""
        try:
            llm_response = llm.invoke(prompt)
            raw_code = llm_response.content
            cleaned_code = clean_python_code(raw_code)
            if not cleaned_code:
                return {
                    "generated_code": None,
                    "error_message": "LLM generated empty code.",
                }
            return {"generated_code": cleaned_code, "error_message": None}
        except Exception as e:
            return {
                "generated_code": None,
                "error_message": f"LLM code generation failed: {str(e)}",
            }

    def execute_python_code_node(state: GraphState):
        typer.echo("Executing Python code...")
        code_to_execute = state.get("generated_code")
        if not code_to_execute:
            return {
                "code_execution_output": None,
                "error_message": state.get(
                    "error_message",
                    "No code to execute due to prior error or empty generation.",
                ),
            }

        try:
            execution_output = python_repl.run(code_to_execute)
            return {"code_execution_output": execution_output, "error_message": None}
        except Exception as e:
            error_detail = f"Python code execution failed: {str(e)}. Code was:\n---\n{code_to_execute}\n---"
            return {"code_execution_output": None, "error_message": error_detail}

    workflow = StateGraph(GraphState)
    workflow.add_node("code_generator", generate_python_code_node)
    workflow.add_node("code_executor", execute_python_code_node)

    workflow.set_entry_point("code_generator")
    workflow.add_edge("code_generator", "code_executor")
    workflow.add_edge("code_executor", END)

    graph_runnable = workflow.compile()

    initial_state = GraphState(
        question=question,
        csv_path=config.CSV_PATH,
        column_names=column_names,
        column_examples_str=column_examples_str,
        generated_code=None,
        code_execution_output=None,
        error_message=None,
    )

    final_state = graph_runnable.invoke(initial_state, {"recursion_limit": 5})

    if final_state.get("error_message"):
        typer.echo(f"An error occurred: {final_state['error_message']}")
    elif final_state.get("code_execution_output"):
        typer.echo("\nExecution Result:")
        typer.echo(final_state["code_execution_output"])
    else:
        typer.echo("No output was generated, and no specific error message was found.")


if __name__ == "__main__":
    app()
