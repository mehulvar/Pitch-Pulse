import traceback

from fastapi import FastAPI

app = FastAPI()  # tiny app that starts instantly


@app.get("/api/ping")
def ping():
    # If this answers, the function itself boots fine on Vercel
    return {"ok": True}


@app.get("/api/debug")
def debug():
    # Load the real app only when asked, so a crash here can't take down /api/ping
    try:
        import pitch_app
        return {"started": True, "teams": len(pitch_app.league)}
    except Exception:
        return {"started": False, "error": traceback.format_exc().splitlines()}