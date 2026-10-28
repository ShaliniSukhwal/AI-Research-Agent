
from typing import Annotated
from typing_extensions import TypedDict
import time

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages

from tools import calculator
from search_tool import web_search


load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls_count: int


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=1,
    timeout=30,
)

tools = [calculator, web_search]
llm_with_tools = llm.bind_tools(tools)


SYSTEM_PROMPT = (
    "You are an AI research assistant that provides accurate, "
    "recent, source-based answers.\n\n"
    "Never change, reconstruct, or guess a source URL. "
    "Copy the exact URL from the matching search result. "
    "Keep each headline, publication date, and URL together. "
    "If the source URL does not match the story, exclude that story.\n\n"

    "GENERAL RULES:\n"
    "1. For current, latest, or today's news, you MUST use "
    "the web_search tool before answering.\n"
    "2. Never invent headlines, dates, facts, or URLs.\n"
    "3. Prefer news published in the last 24 hours. "
    "Exclude results whose publication date cannot be verified.\n"
    "4. Use actual search results and include source URLs.\n"
    "5. Remove duplicate headlines and stories.\n"
    "6. If there are too few relevant results, provide fewer "
    "headlines instead of adding unrelated news.\n\n"

    "LOCATION RULES:\n"
    "1. For national news in India, focus on India-related "
    "national developments.\n"
    "2. Never assume the user means the United States.\n"
    "3. Distinguish Indian news from international news.\n\n"
    "4. For Indian business news, include only stories directly "
    "about India, Indian companies, Indian markets, or India's economy. "
    "Exclude stories about other countries, such as Indonesia, unless "
    "India is a direct and substantial part of the story.\n"
    "5. Never include an international market story merely because "
    "it discusses investors, stocks, or foreign exchange.\n\n"


    "CATEGORY RULES:\n"
    "1. National news: prioritize government, politics, judiciary, "
    "public policy, defence, and major developments affecting India.\n"
    "2. Business news in India: prioritize Indian companies, "
    "markets, banking, the economy, trade, and business policy.\n"
    "3. Technology news in India: prioritize AI, semiconductors, "
    "cybersecurity, startups, gadgets, software, research, "
    "space technology, and technology policy.\n"
    "4. Sports news: prioritize recent sports events, results, "
    "and developments relevant to the requested region.\n"
    "5. International news: cover worldwide developments.\n"
    "6. Include a story in multiple categories only when it is "
    "genuinely relevant to each category.\n\n"

    "CATEGORY FILTERING AND DUPLICATE RULES:\n"
    "1. Exclude stories that do not directly match the requested category.\n"
    "2. Do not repeat the same story within one answer.\n"
    "3. For a new category, search for relevant stories instead "
    "of reusing earlier results.\n"
    "4. Aim for 5 relevant headlines when enough verified results "
    "are available. Never invent stories to reach this number.\n\n"

    "SEARCH RULES:\n"
    "1. Use a search query matching the requested topic, location, "
    "and time period.\n"
    "2. Review titles, dates, content, and URLs before selecting stories.\n"
    "3. Search-result ranking alone does not verify a story.\n"
    "4. If results are insufficient, clearly state that only a "
    "limited number of recent stories were found.\n\n"

    "ANSWER FORMAT:\n"
    "1. Start with a short introduction naming the requested category "
    "and region.\n"
    "2. Present each story with a clear headline and concise summary.\n"
    "3. Use the exact title, publication date, and URL from the same "
    "supplied search result.\n"
    "4. Never reuse one result's URL for another headline.\n"
    "5. Never invent, modify, reconstruct, or guess URLs.\n"
    "6. If a source URL cannot be matched confidently to its headline, "
    "exclude that story.\n"
    "7. Show the publication date for every story.\n"
    "8. Clearly communicate uncertainty when sources do not establish "
    "a fact."
    )


def research_node(state: AgentState):
    print(">>> Gemini call started")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ] + state["messages"]

    if state["tool_calls_count"] >= 2:
        response = llm.invoke(messages)
    else:
        response = llm_with_tools.invoke(messages)

    print(">>> Gemini call finished")
    print("Tool calls:", getattr(response, "tool_calls", None))
    print("Response content:", repr(response.content))

    return {"messages": [response]}


def should_continue(state: AgentState):
    last_message = state["messages"][-1]

    if last_message.tool_calls and state["tool_calls_count"] < 2:
        return "tools"

    return END


def after_tools(state: AgentState):
    return {
        "tool_calls_count": state["tool_calls_count"] + 1
    }


builder = StateGraph(AgentState)

builder.add_node("research", research_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("after_tools", after_tools)

builder.add_edge(START, "research")

builder.add_conditional_edges(
    "research",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

builder.add_edge("tools", "after_tools")
builder.add_edge("after_tools", "research")

graph = builder.compile()


def research(question):
    start_time = time.time()
    print(">>> Graph started")
    print(">>> Question:", question)

    result = graph.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "tool_calls_count": 0,
        }
    )

    print(
        f">>> Total graph time: "
        f"{time.time() - start_time:.2f} seconds"
    )

    last_message = result["messages"][-1]

    print(
        "DEBUG - Last message type:",
        type(last_message).__name__,
    )
    print(
        "DEBUG - Tool calls:",
        getattr(last_message, "tool_calls", None),
    )
    print("DEBUG - Raw content:", repr(last_message.content))

    answer = last_message.content

    if isinstance(answer, list):
        answer = "\n".join(
            item.get("text", "")
            for item in answer
            if isinstance(item, dict) and item.get("type") == "text"
        )

    if not answer or not str(answer).strip():
        print("WARNING - Empty final answer returned by the model.")

    return answer
