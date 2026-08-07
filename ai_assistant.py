import os
import time
import sqlalchemy
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)
MODEL_NAME = "llama-3.3-70b-versatile"

SCHEMA = """
Table name: cleaned_economic_data

Columns:
  date                 DATE  : the month of the observation (e.g. 2022-01-01)
  cpi                  FLOAT : Consumer Price Index - a measure of inflation
  unemployment         FLOAT : unemployment rate, as a percentage
  fed_funds            FLOAT : the Federal Reserve's interest rate, as a percentage
  gdp                  FLOAT : Gross Domestic Product, in billions of dollars
  consumer_sentiment   FLOAT : consumer sentiment index (may be NULL for early dates)
"""


def get_db_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return sqlalchemy.create_engine(url)


def call_groq_with_retry(prompt, max_retries=3):
    """
    Handles retries if the remote inference server hits a high traffic rate limit.
    """
    delay = 1
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq call failed on attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
    raise Exception("Groq API engine unavailable after retries")


def question_to_sql(question):
    """Ask the AI to turn the question into SQL, using only the schema above."""
    prompt = f"""
You are a SQL expert. Convert the user's question into a PostgreSQL query.

Database schema:
{SCHEMA}

Rules:
1. Return ONLY the SQL query - no explanation, no markdown, no backticks.
2. LIMIT results to 10 rows maximum.
3. If the question cannot be answered from this schema, return exactly: CANNOT_ANSWER

User question: {question}

SQL query:"""
    response_text = call_groq_with_retry(prompt)
    sql = response_text.strip().replace("```sql", "").replace("```", "").strip()
    logger.info(f"Generated SQL: {sql}")
    return sql


def run_sql(sql, engine):
    try:
        import pandas as pd
        df = pd.read_sql(sql, engine)
        if df.empty:
            return "NO_ROWS"
        return df.to_string(index=False)
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        return "SQL_ERROR"


def result_to_answer(question, sql_result):
    """Turn the raw query result into one or two plain English sentences."""
    prompt = f"""
The user asked: "{question}"
The database returned this result:
{sql_result}

Write a direct, natural response answering the question based strictly on these numbers.
NEVER use bullet points or bold markers. Use pure unformatted text only.
Include the specific numbers. Do not mention SQL, databases, or rows.
"""
    return call_groq_with_retry(prompt).strip()


def ask(question, engine):
    """Full pipeline: question -> SQL -> real result -> English answer."""
    logger.info(f"Question: {question}")

    sql = question_to_sql(question)
    if "CANNOT_ANSWER" in sql:
        return "I can't answer that from this data - I only have CPI, unemployment, fed funds rate, GDP, and consumer sentiment by month."

    result = run_sql(sql, engine)
    if result == "NO_ROWS":
        return "I didn't find any data for that specific date or range."
    if result == "SQL_ERROR":
        return "I had trouble running that query. Try asking about a specific month, year, or one of: inflation, unemployment, interest rates, GDP, or consumer sentiment."

    return result_to_answer(question, result)


def run_chat():
    print("\n" + "=" * 50)
    print("US ECONOMIC INDICATORS AI ASSISTANT (GROQ POWERED)")
    print("Ask about CPI, unemployment, fed funds, GDP, sentiment")
    print("Type 'quit' to exit")
    print("=" * 50 + "\n")

    engine = get_db_engine()
    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if not question:
            continue
        print("Assistant: thinking...")
        answer = ask(question, engine)
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    run_chat()
    
