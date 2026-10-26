import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv
from google import genai

load_dotenv()
key = os.getenv("GEMINI_API_KEY")
print("Key loaded:", bool(key), "length:", len(key or ""))

# Test 1: a plain web request with no Google SDK. This checks the network and the key
start = time.time()
try:
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                     params={"key": key}, timeout=15)
    print(f"Test 1: status {r.status_code} in {time.time() - start:.1f}s")
    if r.status_code != 200:
        print(r.text[:300])
except Exception as e:
    print(f"Test 1 failed after {time.time() - start:.1f}s: {type(e).__name__}")

# Test 2: the older generate_content method instead of the Interactions API
client = genai.Client()
pool = ThreadPoolExecutor(max_workers=1)
start = time.time()
try:
    future = pool.submit(lambda: client.models.generate_content(
        model="gemini-3.5-flash", contents="Say hello in five words"))
    print(f"Test 2: {time.time() - start:.1f}s -> {future.result(timeout=30).text}")
except Exception as e:
    print(f"Test 2 failed after {time.time() - start:.1f}s: {type(e).__name__}: {e}")

os._exit(0)  # force exit, because a timed-out call can leave a background thread hanging