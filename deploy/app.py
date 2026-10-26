"""Diagnostic entrypoint: starts instantly, then tests the heavy parts one at a time."""
import glob
import importlib
import platform
import sys
import traceback

from fastapi import FastAPI

app = FastAPI()  # tiny app, so the function itself can always start

LIBS = ["numpy", "pandas", "scipy", "sklearn", "joblib", "lightgbm", "google.genai"]


@app.get("/api/ping")
def ping():
    """If this answers, the function itself starts fine."""
    return {"ok": True, "python": sys.version.split()[0], "platform": platform.platform()}


@app.get("/api/import/{name}")
def try_import(name: str):
    """Import one library and report its version or the error.
    One library per request, so a crash in one can't hide the others."""
    if name not in LIBS:
        return {"error": f"pick one of {LIBS}"}
    try:
        module = importlib.import_module(name)
        return {"name": name, "ok": True, "version": getattr(module, "__version__", "unknown")}
    except BaseException:
        return {"name": name, "ok": False, "error": traceback.format_exc().splitlines()[-4:]}


@app.get("/api/system")
def system():
    """Is the OpenMP library that LightGBM needs on this server?"""
    found = []
    for pattern in ["/usr/lib*/libgomp*", "/lib*/libgomp*", "/usr/lib/*/libgomp*"]:
        found += glob.glob(pattern)
    return {"libgomp_on_system": found}


@app.get("/api/debug")
def debug():
    """Load the real app and report whether it starts."""
    try:
        import pitch_app
        return {"started": True, "teams": len(pitch_app.league)}
    except BaseException:
        return {"started": False, "error": traceback.format_exc().splitlines()[-8:]}