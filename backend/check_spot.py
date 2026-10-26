"""Compare our xG with what actually happened for shots from one spot."""
import joblib

from features import load_shots, score_shots

shots = load_shots()
shots["our_xg"] = score_shots(shots, joblib.load("models/xg_model.joblib"))

# Central open-play right-foot shots after a pass, 15 to 18 meters from goal.
# X is a fraction of the pitch length, so 0.83 to 0.86 is about 15 to 18 meters out
spot = shots[
    shots.X.between(0.83, 0.86) & shots.Y.between(0.42, 0.58)
    & (shots.situation == "OpenPlay") & (shots.shotType == "RightFoot")
    & (shots.lastAction == "Pass")
]
print("Shots like this:", len(spot))
print("Actually scored:", round(spot.goal.mean(), 3))
print("Understat xG:   ", round(spot.xG.mean(), 3))
print("Our xG:         ", round(spot.our_xg.mean(), 3))