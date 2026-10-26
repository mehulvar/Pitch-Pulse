import os
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()
pool = ThreadPoolExecutor(max_workers=1)

# Quotas are per model, so find one that still has free requests today
for model in ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash"]:
    start = time.time()
    try:
        future = pool.submit(lambda m=model: client.models.generate_content(
            model=m, contents="Say hello in five words"))
        print(f"{model}: OK in {time.time() - start:.1f}s -> {future.result(timeout=20).text.strip()}")
    except Exception as e:
        print(f"{model}: FAILED -> {str(e)[:120]}")

os._exit(0)