
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    max_retries=0,
)

start = time.time()
print("Testing Gemini...")

response = llm.invoke("Reply in one short sentence: What is AI?")

print("Time:", round(time.time() - start, 2), "seconds")
print("Answer:", response.content)
