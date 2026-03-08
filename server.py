"""
Jacob's Ladder - Backend Server
Evaluates player responses using Claude API.
"""

import os
import json
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a wise, kind Sunday School teacher evaluating a 12-year-old boy's response to a moral dilemma. This is for a game called "Jacob's Ladder" for the Deacon's quorum in The Church of Jesus Christ of Latter-day Saints.

You must return ONLY a valid JSON object with NO markdown formatting, NO code fences, and NO extra text. Just raw JSON.

The JSON must have these fields:
{
  "score": <integer from -3 to 3>,
  "feedback": "<2-3 sentences>",
  "scripture": "<reference like 'Matthew 5:44'>",
  "scripture_text": "<actual verse text>"
}

Scoring guide:
+3: Exceptionally Christlike - shows real sacrifice, empathy, courage, and moral reasoning beyond their years
+2: Good and thoughtful - kind, honest, and considers others' feelings
+1: Decent - tries to do right but takes the easy path or misses the deeper issue
 0: Neutral - avoidant, vague, or doesn't really address the dilemma
-1: Somewhat selfish - prioritizes self-interest or peer pressure over doing right
-2: Unkind or dishonest - actively chooses to hurt, deceive, or go along with wrong
-3: Cruel or deeply wrong - bullying, stealing, or completely lacking empathy

For feedback: Be encouraging but honest. Speak directly to them as "you." Reference gospel principles when relevant. If the answer is bad, gently explain why without being harsh.

For scripture: Pick a verse that genuinely relates to the specific situation. Use Book of Mormon, Bible, or D&C references as appropriate.

Be fair but not too generous. Most thoughtful answers should score +1 or +2. Reserve +3 for truly exceptional moral reasoning. Don't hesitate to give negative scores for genuinely bad choices."""


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/sprites/<path:filename>")
def sprites(filename):
    return send_from_directory("sprites", filename)


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    data = request.json
    scenario = data.get("scenario", "")
    response_text = data.get("response", "")
    question_num = data.get("questionNumber", 1)

    user_prompt = f"""SCENARIO (Question {question_num} of 10):
{scenario}

THE BOY'S RESPONSE:
{response_text}

Evaluate this response and return ONLY the JSON object."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        raw = response.content[0].text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        result = json.loads(raw)

        # Validate and clamp score
        score = int(result.get("score", 0))
        score = max(-3, min(3, score))
        result["score"] = score

        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify(fallback_scoring(response_text))
    except Exception as e:
        print(f"LLM error: {e}")
        return jsonify(fallback_scoring(response_text))


def fallback_scoring(response_text):
    """Basic keyword scoring when LLM is unavailable."""
    lower = response_text.lower()
    score = 0

    positive = ["help", "kind", "honest", "pray", "forgive", "friend", "love",
                 "sorry", "talk", "listen", "truth", "right", "serve", "include"]
    negative = ["ignore", "steal", "lie", "laugh", "mean", "hit", "punch",
                "fight", "revenge", "mock", "bully"]

    pos_count = sum(1 for w in positive if w in lower)
    neg_count = sum(1 for w in negative if w in lower)
    score = min(2, pos_count) - min(2, neg_count)

    if len(response_text) > 150:
        score = min(3, score + 1)

    return {
        "score": score,
        "feedback": "Your response shows thought and care. (Note: AI evaluation was unavailable, so this is simplified scoring.)",
        "scripture": "Proverbs 3:5-6",
        "scripture_text": "Trust in the Lord with all thine heart; and lean not unto thine own understanding. In all thy ways acknowledge him, and he shall direct thy paths.",
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True)
