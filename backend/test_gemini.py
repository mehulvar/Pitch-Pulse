import os
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from google import genai

load_dotenv()  # loads GEMINI_API_KEY from .env
client = genai.Client()
pool = ThreadPoolExecutor(max_workers=2)

# Send a tiny prompt to each model and time it. Give up after 30 seconds
for model in ["gemini-3.5-flash", "gemini-4-argon"]:
    start = time.time()
    try:
        future = pool.submit(lambda m=model: client.interactions.create(
            model=m, input="Say hello in five words"))
        text = future.result(timeout=30).output_text
        print(f"{model}: {time.time() - start:.1f}s -> {text}")
    except Exception as e:
        print(f"{model}: failed after {time.time() - start:.1f}s -> {type(e).__name__}: {e}")

os._exit(0)  # force exit, because a timed-out call can leave a background thread hanging