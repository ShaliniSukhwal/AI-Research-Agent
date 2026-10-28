
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.tools import tool
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse
import os
import requests
import re

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

INDIA_DOMAINS = [
    "reuters.com",
    "thehindu.com",
    "indianexpress.com",
    "hindustantimes.com",
    "ndtv.com",
    "pti.in",
    "pib.gov.in",
]

BUSINESS_KEYWORDS = {
    "business", "economy", "company", "companies", "corporate",
    "market", "stock", "stocks", "ipo", "sebi", "bank", "banking",
    "finance", "financial", "trade", "investment", "profit",
    "revenue", "earnings", "merger", "acquisition", "startup",
    "industry", "inflation", "gdp", "rupee", "rbi", "exports",
    "imports", "tax", "budget", "shares", "investors"
}

TECH_KEYWORDS = {
    "technology", "tech", "artificial intelligence", "ai",
    "generative ai", "machine learning", "semiconductor",
    "semiconductors", "chip", "chips", "cybersecurity",
    "software", "gadget", "gadgets", "startup", "startups",
    "digital", "robotics", "telecom", "6g", "5g",
    "google", "microsoft", "apple", "openai", "nvidia",
    "isro", "space technology", "data centre", "data center",
    "cloud computing", "cyber attack", "smartphone",
    "electronics", "electronic goods", "technology policy",
    "innovation", "research and development", "deep tech",
    "information technology", "it services", "data privacy",
    "data protection", "app launch", "launches ai"
}

NATIONAL_KEYWORDS = {
    "india", "indian", "government", "ministry", "parliament",
    "supreme court", "policy", "minister", "cabinet", "election",
    "defence", "defense", "drdo", "judiciary", "pib", "sebi",
    "rbi", "lok sabha", "rajya sabha", "bill", "legislation"
}


def parse_published_date(value):
    if not value:
        return None

    try:
        published = parsedate_to_datetime(value)
    except (ValueError, TypeError, OverflowError):
        try:
            published = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except (ValueError, TypeError, OverflowError):
            return None

    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)

    return published.astimezone(timezone.utc)


def detect_category(query):
    q = query.lower()

    if any(term in q for term in [
        "business news", "business headlines", "economy news",
        "economic news", "corporate news", "stock market news"
    ]):
        return "business"

    if any(term in q for term in [
        "technology news", "tech news", "ai news",
        "semiconductor news", "startup news"
    ]):
        return "technology"

    if any(term in q for term in [
        "sports news", "sport news", "cricket news", "football news"
    ]):
        return "sports"

    if any(term in q for term in [
        "national news", "india news", "indian news",
        "government policy", "government news"
    ]):
        return "national"

    return "general"


def normalize_text(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()



def is_relevant(title, content, category):
    title_text = title.lower()
    content_text = content.lower()[:700]

    if category == "business":
        business_terms = {
            "business", "economy", "company", "companies", "market",
            "stock", "ipo", "bank", "banking", "finance", "trade",
            "investment", "profit", "revenue", "earnings", "merger",
            "acquisition", "startup", "industry", "inflation", "gdp",
            "rupee", "rbi", "sebi", "shares", "investor", "exports",
            "imports", "tax", "budget", "sensex", "nifty", "financial"
        }
        india_terms = {
            "india", "indian", "rupee", "sensex", "nifty",
            "sebi", "rbi", "bse", "nse"
        }

        has_business_topic = any(
            term in title_text for term in business_terms
        )
        has_india_context = any(
            term in title_text or term in content_text
            for term in india_terms
        )
        return has_business_topic and has_india_context

    if category == "technology":
        technology_terms = {
            "technology", "tech", "artificial intelligence",
            "semiconductor", "chip", "cybersecurity", "software",
            "gadget", "startup", "digital", "robotics", "telecom",
            "6g", "5g", "google", "microsoft", "apple", "openai",
            "nvidia", "isro", "space technology", "data centre",
            "data center", "cloud computing", "smartphone",
            "electronics", "deep tech", "machine learning",
            "generative ai", "it services"
        }
        text_to_check = title_text + " " + content_text
        return any(term in text_to_check for term in technology_terms)

    if category == "national":
        return any(term in title_text for term in NATIONAL_KEYWORDS)

    if category == "sports":
        return any(term in title_text for term in {
            "cricket", "football", "hockey", "tennis", "badminton",
            "olympics", "sports", "sport"
        })

    return True


def url_matches_title(title, url):
    """
    Reject obvious title/source mismatches when the URL path
    clearly points to an unrelated subject.
    This is a heuristic, not full article verification.
    """
    parsed = urlparse(url)

    if parsed.scheme != "https" or not parsed.netloc:
        return False

    path_text = normalize_text(parsed.path)

    # If the URL contains a clear subject unrelated to the title,
    # reject it. This catches some obvious mismatched URLs.
    title_words = set(normalize_text(title).split())
    path_words = set(path_text.split())

    meaningful_title_words = {
        word for word in title_words
        if len(word) >= 5
        and word not in {
            "india", "indian", "latest", "report", "reports",
            "news", "today", "over", "after", "from", "with",
            "says", "warns", "plans", "could", "would"
        }
    }

    if not meaningful_title_words:
        return True

    # A URL may use a shorter or different slug, so this check
    # only rejects a strong mismatch rather than requiring every
    # title word to appear in the URL.
    overlap = meaningful_title_words.intersection(path_words)

    return bool(overlap)


@tool
def web_search(query: str) -> str:
    """Search for recent news and return dated, relevant source results."""

    try:
        category = detect_category(query)

        search_query = query

        if category == "business":
            search_query += (
                " India Indian companies economy banking markets"
            )
        elif category == "technology":
            search_query += (
                " India technology AI semiconductors startups"
            )
        elif category == "national":
            search_query += " India government policy"

        response = tavily.search(
            query=search_query,
            search_depth="advanced",
            topic="news",
            time_range="day",
            max_results=20,
            include_domains=INDIA_DOMAINS,
        )

        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

        results = []
        seen_urls = set()
        seen_titles = set()

        for item in response.get("results", []):
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            content = (item.get("content") or "").strip()
            published_date = item.get("published_date")

            if not title or not url or not published_date:
                continue

            published = parse_published_date(published_date)

            if published is None:
                continue

            if published < cutoff or published > datetime.now(timezone.utc):
                continue

            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower().removeprefix("www.")

            if not any(
                domain == allowed or domain.endswith("." + allowed)
                for allowed in INDIA_DOMAINS
            ):
                continue

            # if not url_matches_title(title, url):
#     print("Rejected possible title/URL mismatch:", title, url)
#     continue

            print("Checking story:", title, "| Date:", published_date)
            if not is_relevant(title, content, category):
             print("Rejected by relevance filter:", title)
             continue

            normalized_title = normalize_text(title)
            normalized_url = url.split("?")[0].rstrip("/").lower()

            if normalized_title in seen_titles or normalized_url in seen_urls:
                continue

            seen_titles.add(normalized_title)
            seen_urls.add(normalized_url)

            results.append({
                "title": title,
                "date": published.isoformat(),
                "url": url,
                "content": content[:1800],
            })

        print("Search query:", search_query)
        print("Detected category:", category)
        print("Filtered results:", len(results))

        for item in results:
            print("Title:", item["title"])
            print("URL:", item["url"])

        if not results:
            return (
                "No relevant, dated results passed the filters "
                "for the last 24 hours. Do not invent headlines, "
                "publication dates, article details, or URLs. "
                "Report that no verified results were available."
            )

        formatted_results = []

        for item in results:
            formatted_results.append(
                f"Title: {item['title']}\n"
                f"Published date: {item['date']}\n"
                f"URL: {item['url']}\n"
                f"Content: {item['content']}"
            )

        return (
            f"Requested category: {category}\n"
            f"Search results for: {query}\n\n"
            "Use only these supplied search results. Do not create "
            "new stories or change a source URL. Never attach a URL "
            "to a different headline. Do not claim that an article "
            "was independently verified if only its search result "
            "was checked. If fewer relevant stories are available, "
            "return fewer stories.\n\n"
            + "\n---\n".join(formatted_results)
        )

    except (requests.exceptions.RequestException, ConnectionError) as error:
        print("Web search connection error:", error)
        return "Web search is temporarily unavailable. Current news was not verified."

    except Exception as error:
        print("Web search error:", repr(error))
        return "Web search failed temporarily. Check the terminal error and try again."


if __name__ == "__main__":
    result = web_search.invoke(
        {"query": "Today's business news in India"}
    )
    print(result)
