import json, os, time
import pandas as pd
from src.models.fusion import predict

def main(n=15, delay=8):
    labels = pd.read_csv("data/labels.csv")
    notes_pids = set(pd.read_csv("data/notes.csv").patient_id)
    vitals_pids = set(pd.read_csv("data/vitals.csv").patient_id)
    usable = [p for p in labels.patient_id if p in notes_pids and p in vitals_pids]
    pids = usable[:n]

    os.makedirs("results", exist_ok=True)

    # resume if we already have some saved
    preds = {}
    if os.path.exists("results/predictions.json"):
        with open("results/predictions.json") as f:
            preds = json.load(f)

    for i, pid in enumerate(pids):
        if str(pid) in preds:
            print(f"[{i+1}/{len(pids)}] patient {pid} already done, skipping")
            continue
        try:
            preds[int(pid)] = predict(int(pid))
            # SAVE AFTER EVERY SUCCESS so we never lose progress
            with open("results/predictions.json", "w") as f:
                json.dump(preds, f, indent=2)
            print(f"[{i+1}/{len(pids)}] patient {pid} done & saved")
        except Exception as e:
            print(f"[{i+1}/{len(pids)}] patient {pid} FAILED: {e}")
        time.sleep(delay)

    # save the vitals for whatever patients we successfully got
    vitals = pd.read_csv("data/vitals.csv")
    got = [int(p) for p in preds.keys()]
    sub = vitals[vitals.patient_id.isin(got)]
    sub.to_json("results/vitals_demo.json", orient="records")

    print(f"\nTotal saved: {len(preds)} predictions in results/predictions.json")

if __name__ == "__main__":
    main()