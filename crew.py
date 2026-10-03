"""
AI News LinkedIn Crew: a 3-agent CrewAI pipeline.

Researcher (with live news search) -> Writer -> Editor, run sequentially.
The Researcher finds this week's most important developments in Generative AI,
LLMs and agentic AI; the Writer turns them into a LinkedIn post; the Editor
checks every claim against the sources. The final post is saved to output/.
"""
import os
import sys
import time
from datetime import date, datetime
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # load keys and settings before CrewAI starts

from crewai import Agent, Crew, LLM, Process, Task, TaskOutput  # noqa: E402
from crewai.tools import tool  # noqa: E402
from tavily import TavilyClient  # noqa: E402

for key in ("OPENAI_API_KEY", "TAVILY_API_KEY"):
    if not os.getenv(key):
        sys.exit(f"{key} not found. Add it to your .env file (see .env.example).")

# The Researcher (choosing stories) and the Editor (catching errors) do the judgment-heavy work,
# so they use the stronger model. The Writer only drafts from the notes, so the cheaper mini model is enough.
strong_llm = LLM(model="openai/gpt-4.1", temperature=0.2)
llm = LLM(model="openai/gpt-4.1-mini", temperature=0.3)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


MIN_SEARCHES = 5
MAX_POST_CHARS = 1500
search_count = 0  # how many searches the Researcher has run in this crew run


# ---------------------------------------------------------------- tools
@tool("Search recent AI news")
def search_ai_news(query: str) -> str:
    """Search news articles from the last 7 days. Input is a short search query,
    e.g. 'new LLM model launch' or 'AI startup funding round'. Returns titles,
    publication dates, URLs and summaries."""
    global search_count
    search_count += 1
    results = tavily.search(query=query, topic="news", days=7, max_results=5, search_depth="advanced")
    items = []
    for r in results.get("results", []):
        items.append(
            f"Title: {r.get('title')}\n"
            f"Published: {r.get('published_date', 'unknown')}\n"
            f"URL: {r.get('url')}\n"
            f"Summary: {r.get('content')}"
        )
    return "\n\n".join(items) or "No recent results found."


# ---------------------------------------------------------------- agents
researcher = Agent(
    role="AI News Researcher",
    goal="Find the most significant developments of the past week in {topic}, with verified sources",
    backstory=(
        "You track the AI industry for a professional audience. You rely only on the search tool's results, "
        "never on memory, and you never invent model names, numbers, companies or dates. Every point you report "
        "includes its source URL and publication date."
    ),
    tools=[search_ai_news],
    llm=strong_llm,
    max_iter=10,
    verbose=True,
)

writer = Agent(
    role="LinkedIn AI Content Creator",
    goal="Share the week's biggest AI news in a personal, conversational post that people want to reply to",
    backstory=(
        "You are an AI engineer who posts a weekly AI roundup on LinkedIn for your network. You write the way "
        "you talk: in the first person, directly to your readers ('you'), with your own short reactions to each "
        "story (what surprised you, what you would try, what you are watching). You never sound like a news "
        "summary or a press release, and you avoid hype and buzzwords."
    ),
    llm=llm,
    verbose=True,
)

editor = Agent(
    role="Fact-Checking Editor",
    goal="Make the post accurate, concise and ready to publish",
    backstory=(
        "You check every claim in the draft against the research notes and remove anything they do not support. "
        "You respect the writer's voice: you make the smallest changes needed and never rewrite a post into "
        "a dry summary."
    ),
    llm=strong_llm,
    verbose=True,
)

# ---------------------------------------------------------------- guardrails
# Prompts are instructions the model can ignore; guardrails are Python checks it cannot.
# If a check fails, CrewAI sends the feedback to the agent and it redoes the task.
# After 3 failed retries the output is accepted anyway, so a run never crashes over this.
GUARDRAIL_RETRIES = 3
guardrail_failures = {"research": 0, "edit": 0}


def enough_searches(output: TaskOutput) -> tuple[bool, Any]:
    if search_count >= MIN_SEARCHES or guardrail_failures["research"] >= GUARDRAIL_RETRIES:
        return True, output.raw
    guardrail_failures["research"] += 1
    return False, (
        f"You have run only {search_count} searches, but at least {MIN_SEARCHES} are required. Run more searches "
        "for the categories you have not covered yet, then redo the scored longlist and the top 3."
    )


def post_length_ok(output: TaskOutput) -> tuple[bool, Any]:
    post = output.raw.split("Sources:")[0].strip()
    if len(post) <= MAX_POST_CHARS or guardrail_failures["edit"] >= GUARDRAIL_RETRIES:
        return True, output.raw
    guardrail_failures["edit"] += 1
    return False, (
        f"The post (without the Sources list) is {len(post)} characters. It must be at most {MAX_POST_CHARS}, so "
        f"cut about {len(post) - MAX_POST_CHARS + 50} characters. Shorten the factual sentences first; keep the "
        "hook, the 'My take' lines, the closing question and the hashtags. Return the full post and the Sources list."
    )


# ---------------------------------------------------------------- tasks
research_task = Task(
    description=(
        "Today is {today}. Find the 3 most significant developments of the past 7 days in {topic}.\n\n"
        "Step 1 - Search widely. Run at least 5 searches, including: new model launches; AI developer "
        "conference, keynote or launch-event announcements from any company (search for the category, not a "
        "specific event name); new AI agent platforms or "
        "developer tools; major AI funding rounds; AI safety. If a promising story's summary has no concrete "
        "facts (only background), run a follow-up search using its name before you drop it.\n\n"
        "Step 2 - Check freshness. A page's 'Published' date can be recent even when it republishes old news, "
        "so check the dates mentioned inside the summary and reject any item whose event happened more than "
        "7 days ago.\n\n"
        "Step 3 - Score a longlist. List 6-8 candidate stories. Skip industry trend reports, surveys, "
        "partnerships, customer-service chatbots and opinion pieces. Score each candidate from 1 to 5 on: "
        "(A) Usable now: does it change what AI engineers can build or use today (a shipped model, API, SDK, "
        "agent platform or developer feature scores high; an announcement of something not yet available, a "
        "delay or a funding round scores lower). "
        "(B) Scale: size of the launch, money or risk. "
        "(C) Coverage: how many different outlets in your results covered it. "
        "Add the three scores for a total out of 15.\n\n"
        "Step 4 - Choose. Pick the 3 highest-scoring stories, with at least one new model, agent or "
        "developer-platform launch and no more than 2 from the same company.\n\n"
        "Accuracy rules: report only what the sources say; do not add reasons, purposes or motives the source "
        "does not state. If a source cites unnamed people or says 'reportedly' or 'in talks', keep that "
        "wording. If you compare two numbers (for example, valuations), calculate the change exactly."
    ),
    expected_output=(
        "First, the scored longlist as a table (story, A, B, C, total). "
        "Then the 3 chosen developments, each with: a headline, 2-3 factual sentences (what happened, key "
        "numbers or capabilities), short verbatim quotes from the search results that support every one of "
        "those facts (labelled 'Evidence:'), the source URL, the publication date and the date the event "
        "happened. Use only facts found in the search results. If a fact comes from a different article than "
        "the main source, list that article's URL too."
    ),
    agent=researcher,
    guardrail=enough_searches,
    guardrail_max_retries=GUARDRAIL_RETRIES,
)

write_task = Task(
    description=(
        "Using only the 3 chosen developments in the research notes (ignore the longlist), write this week's "
        "LinkedIn AI roundup in your own voice, as a creator talking to your followers. "
        "Open with a personal, specific one-line hook about the first (highest-scoring) development, in your own "
        "words: a surprising fact, a strong opinion or a question. Do not start with 'I did not expect' and do not "
        "make a general statement about AI. Follow it with one short line that tells readers what you are sharing. "
        "Present the developments in the order the notes rank them. "
        "Give each development its own short paragraph, numbered or marked with an arrow: what happened, with "
        "concrete names and numbers, then your own one-sentence reaction or takeaway in the first person "
        "('My take: ...', 'What I'd watch: ...'). Every development must have its own reaction line. "
        "Address the reader as 'you' at least once. "
        "Use short sentences and line breaks so the post is easy to read on a phone. "
        "Do not invent personal experiences: reactions are opinions, not claims that you used a product. "
        "Avoid phrases such as 'game-changer', 'exciting times', 'this highlights', 'in today's fast-paced world', "
        "'these developments emphasize', 'where the rubber meets the road' and other clichés. "
        "Keep the source's hedging: if the notes say 'reportedly' or 'in talks', say so in the post. "
        "End by asking your readers one specific question (only one) about the news, then 3-5 hashtags."
    ),
    expected_output="A complete, conversational LinkedIn post draft in plain text, written in the first person.",
    agent=writer,
    context=[research_task],
)

edit_task = Task(
    description=(
        "Fact-check the draft sentence by sentence against the 'Evidence:' quotes in the research notes. "
        "Correct or remove any claim, number, comparison, purpose or motive the quotes do not directly support, "
        "restore hedging such as 'reportedly' where the source uses it, and recalculate any comparison of "
        "numbers. Use quotation marks only for words the source attributes to that person; a reporter's "
        "description is not a quote. Treat any statement in the research notes that is not backed by an "
        "'Evidence:' quote as unverified and remove it from the post. Make sure the Sources list includes every "
        "URL a fact came from. Flag any development whose event date is older than 7 days. "
        "Keep the writer's opening hook, conversational first-person voice, personal reactions ('My take' "
        "lines), line breaks, closing question and hashtags. "
        "Change a sentence only if it is inaccurate, unclear or wordy. Do not add a concluding or summary "
        "sentence, and do not use phrases like 'these developments emphasize' or 'this highlights'. "
        "Keep the post under 1,500 characters (a Python check will measure it); if you need to cut, shorten the "
        "factual sentences first, never the hook, the 'My take' lines or the question. After the post, add a 'Sources:' list with the URL of each "
        "development."
    ),
    expected_output=(
        "The final LinkedIn post, starting with the writer's hook and ending with the writer's question and "
        "hashtags, followed by a Sources list."
    ),
    agent=editor,
    context=[research_task, write_task],
    guardrail=post_length_ok,
    guardrail_max_retries=GUARDRAIL_RETRIES,
)

crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, write_task, edit_task],
    process=Process.sequential,
    verbose=True,
)


def main():
    topic = " ".join(sys.argv[1:]) or "Generative AI, large language models and agentic AI"
    inputs = {"topic": topic, "today": date.today().strftime("%d %B %Y")}

    # Model APIs are occasionally overloaded (503) or briefly rate-limited (429).
    # These errors are temporary, so wait and retry the run up to 3 times.
    for attempt in range(1, 4):
        try:
            result = crew.kickoff(inputs=inputs)
            break
        except Exception as e:
            temporary = any(code in str(e) for code in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED"))
            if not temporary or attempt == 3:
                raise
            print(f"\nThe model API is busy or rate-limited (attempt {attempt} of 3). Retrying in 60 seconds...\n")
            time.sleep(60)

    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", f"post_{datetime.now():%Y%m%d_%H%M%S}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Weekly AI roundup: {topic}\n\n{result.raw}\n")

    print(f"\nSaved to {path}")

    # Language models are unreliable at counting characters, so check the length in code.
    post = result.raw.split("Sources:")[0].strip()
    print(f"Searches run: {search_count}")
    if len(post) > MAX_POST_CHARS:
        print(f"Note: the post is {len(post)} characters (target: under 1,500). Trim it before publishing.")
    else:
        print(f"Post length: {len(post)} characters.")


if __name__ == "__main__":
    main()