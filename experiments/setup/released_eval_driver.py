"""Run the systems' RELEASED evaluator (LadderSYM evaluate_errors.py, verbatim copy)
on our prediction MIDIs, so the backward-compatible mode can be compared with the
protocol as its authors implemented it. Per-piece means of per-track onset F1:
Track 0 = extra, Track 1 = removed (missed), Track 2 = correct."""
import json, sys, os
sys.path.insert(0, 'run')
import ls_evaluate_errors as EV
gt = json.load(open('run/gt_meta_maestro.json'))
gt_list = [dict(track_id=k, **{kk: v[kk] for kk in ('extra_notes_midi', 'removed_notes_midi', 'correct_notes_midi')}) for k, v in gt.items()]
out = {"_provenance": {"run": "released_eval_driver.py on gilbreth 2026-09-01",
                       "evaluator": "LadderSYM evaluate_errors.py (verbatim), per-piece mean of per-track mir_eval onset F1",
                       "tracks": {"0": "extra", "1": "removed/missed", "2": "correct"}}}
for name in sys.argv[1:]:
    scores, means = EV.evaluate_main("MAESTRO", f"run/out/{name}/full", gt_list)
    out[name] = {k: float(v) for k, v in means.items()}
    out[name]["n_pieces"] = len(scores.get("Onset F1", []))
    print(name, {k: round(v, 4) for k, v in out[name].items() if "F1" in k})
json.dump(out, open('run/results/released_evaluator.json', 'w'), indent=1)
print("wrote run/results/released_evaluator.json")
