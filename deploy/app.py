import traceback

from fastapi import FastAPI

try:
    from main import app  # the real app
except Exception:
    # If the real app crashes while starting, serve the error text instead
    app = FastAPI()
    error_text = traceback.format_exc()

    @app.get("/{path:path}")
    def show_error(path: str):
        return {"error": error_text.splitlines()}