"""Vercel entrypoint. Loads libgomp first so LightGBM can start, then hands requests to the real app."""
import ctypes
import os
import traceback

from fastapi import FastAPI

HERE = os.path.dirname(os.path.abspath(__file__))

# LightGBM needs the OpenMP library, which this server doesn't have. We ship our own copy
try:
    ctypes.CDLL(os.path.join(HERE, "lib", "libgomp.so.1"), mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass  # nothing to do if the library can't be loaded here

app = FastAPI()  # defined at the top level so Vercel can find the entrypoint


@app.get("/api/ping")
def ping():
    return {"ok": True}


try:
    import pitch_app

    app.mount("/", pitch_app.app)  # every other request goes to the real app
except BaseException:
    ERROR = traceback.format_exc().splitlines()[-8:]

    @app.get("/api/{path:path}")
    def show_error(path: str):
        # If the real app can't start, show why instead of a blank 500
        return {"started": False, "error": ERROR}
