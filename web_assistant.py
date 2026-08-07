import os
import time
import logging
import sqlalchemy
import pandas as pd
from flask import Flask, request, render_template_string
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

groq_client = OpenAI(
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

PAGE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>US Economic Indicators Explorer</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 650px; margin: 50px auto; padding: 0 20px; color: #333; }
        input[type=text] { width: 100%; padding: 12px; font-size: 15px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
        button { padding: 11px 24px; margin-top: 12px; font-size: 15px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
        button:hover { background-color: #0056b3; }
        .box { margin-top: 25px; padding: 20px; background: #e8f4ff; border-left: 5px solid #007bff; border-radius: 4px; line-height: 1.6; }
    </style>
</head>
<body>
    <h1>US Economic Indicators Database Assistant</h1>
    <p>Ask plain-English questions about inflation (CPI), unemployment, interest rates, GDP, or consumer sentiment metrics housed in the database.</p>

    <form method="POST">
        <input type="text" name="question" placeholder="e.g., What was the unemployment rate in 2020?" value="{{ question or '' }}" required>
        <button type="submit">Query Database</button>
    </form>

    {% if answer %}
    <div class="box">
        <strong>Response:</strong><br><br>
        {{ answer }}
    </div>
    {% endif %}
</body>
</html>
"""

def get_db_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    return sqlalchemy.create_engine(url)

def call_groq_network_route(prompt, max_retries=3):
    delay = 1
    for attempt in range(1, max_retries + 1):
        try:
            response = groq_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Network route attempt {attempt} failed: {e}")
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
    raise Exception("Groq cloud cluster gateway timed out or refused authentication.")

def run_pipeline(question):
    try:
        sql_prompt = f"""You are a SQL expert. Convert the user's question into a PostgreSQL query.
        Database schema:
        {SCHEMA}
        Rules:
        1. Return ONLY the SQL query - no explanation, no markdown, no backticks.
        2. LIMIT results to 10 rows maximum.
        3. If the question cannot be answered from this schema, return exactly: CANNOT_ANSWER
        User question: {question}
        SQL query:"""

        generated_sql = call_groq_network_route(sql_prompt).strip()
        generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()
        logger.info(f"Successfully Compiled SQL: {generated_sql}")

        if "CANNOT_ANSWER" in generated_sql or not generated_sql.startswith("SELECT"):
            return "I can't find metrics for that specific domain request. I can assist with CPI, unemployment, interest rates, GDP, and consumer sentiment data."

        try:
            engine = get_db_engine()
            df = pd.read_sql(generated_sql, engine)
            if df.empty:
                return "No matching records were located inside the database infrastructure for that timestamp parameter."
            db_result_text = df.to_string(index=False)
        except Exception as e:
            logger.error(f"PostgreSQL Backend processing crash: {e}")
            return "The server encountered a configuration issue while evaluating the database table constraints. Please try rephrasing your timeline targets."

        narrative_prompt = f"""Someone asked: "{question}"
        The database table returned these real numbers:
        {db_result_text}
        Write a direct, natural response answering the question based strictly on these numbers.
        NEVER use bullet points, structural items, or bold markers. Use pure unformatted paragraph text only.
        Include the specific numbers. Do not mention SQL, databases, rows, or data frames."""

        return call_groq_network_route(narrative_prompt).strip()
    except Exception as exc:
        logger.error(f"Groq pipeline error: {exc}")
        return "The assistant could not complete the AI response. The Groq service may be unavailable or timed out. Please try again in a moment."

@app.route("/", methods=["GET", "POST"])
def home():
    question = None
    answer = None
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            try:
                answer = run_pipeline(question)
            except Exception as exc:
                logger.error(f"Web route handler failed: {exc}")
                answer = "The assistant is temporarily unavailable. Please retry after a short moment."
    return render_template_string(PAGE_TEMPLATE, question=question, answer=answer)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
