# web_assistant.py
# Simple web page for the AI assistant - type a question, get an answer,
# instead of using the terminal chat loop.

from flask import Flask, request, render_template_string
from ai_assistant import ask, get_db_engine

app = Flask(__name__)
engine = get_db_engine()

PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>US Economic Indicators Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 60px auto; }
        input[type=text] { width: 100%; padding: 10px; font-size: 16px; }
        button { padding: 10px 20px; margin-top: 10px; font-size: 16px; }
        .answer { margin-top: 20px; padding: 15px; background: #f0f0f0; border-radius: 6px; }
    </style>
</head>
<body>
    <h2>Ask about US Economic Indicators</h2>
    <p>CPI, unemployment, fed funds rate, GDP, consumer sentiment</p>

    <form method="POST">
        <input type="text" name="question" placeholder="e.g. What was unemployment in 2020?"
               value="{{ question or '' }}">
        <br>
        <button type="submit">Ask</button>
    </form>

    {% if answer %}
    <div class="answer"><strong>Answer:</strong> {{ answer }}</div>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    answer = None
    question = None
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            answer = ask(question, engine)
    return render_template_string(PAGE, question=question, answer=answer)


if __name__ == "__main__":
    app.run(debug=True, port=5000)