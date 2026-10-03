# AI News LinkedIn Crew (CrewAI)

A three-agent [CrewAI](https://github.com/crewAIInc/crewAI) pipeline that finds the past week's most important developments in Generative AI, LLMs and agentic AI and turns them into a fact-checked LinkedIn post. Built as part of the Gen AI & Agentic AI Summer School 2026 internship at IIT Jammu.

## How it works

```
Topic → AI News Researcher (live news search) → LinkedIn Content Writer → Fact-Checking Editor → output/post_<timestamp>.md
```

| Agent | Role | Output |
|---|---|---|
| AI News Researcher | Searches news from the last 7 days with a custom Tavily tool and picks the 3 most significant developments, such as model launches, funding rounds and agentic-AI products | Research notes with source URLs and dates |
| LinkedIn Content Writer | Turns the notes into a weekly roundup post with a hook, a short paragraph per development, a closing question and hashtags | Draft post |
| Fact-Checking Editor | Checks every claim against the notes, removes anything unsupported, keeps the post under 1,300 characters and lists the sources | Final post with sources |

- **Process:** sequential. Each task passes its output to the next through `context`, and the Editor receives both the research and the draft.
- **Grounding:** the Researcher is instructed to use only the search results, never the model's memory, so the post reflects current news rather than outdated training data.
- **Model:** OpenAI GPT-4.1 mini through CrewAI's native OpenAI provider (low-cost, billed per token).
- **Output:** the final post is printed in the terminal and saved as a Markdown file in `output/`. Nothing is published automatically.

## Run locally

Requires Python 3.10–3.13.

```bash
git clone https://github.com/eshitanarang/crewai-linkedin-crew.git
cd crewai-linkedin-crew
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your keys:

```
OPENAI_API_KEY=your_openai_api_key     # from platform.openai.com
TAVILY_API_KEY=your_tavily_api_key     # free tier from tavily.com
```

Then run with the default topic, or your own:

```bash
python crew.py
python crew.py "AI agents and agent frameworks"
```

`.env` is listed in `.gitignore`, so the API keys are never committed.

## Tech stack

Python · CrewAI · OpenAI API · Tavily Search API · python-dotenv