#!/usr/bin/env python3
"""verify_shipped.py — mechanical re-verification of every artifact we ship.

Written 2026-07-20 after two avoidable defects reached the cluster:
  * scripts rewritten from Windows carried CRLF line endings, which bash on
    Linux cannot parse — every launched job died in 18 s;
  * a documentation claim ("all tracks named Acoustic Grand Piano") was true
    for one of three model configurations and false for the other two.

Both classes are mechanically detectable. This script is the gate that must be
green before anything is described as shipped. It checks the artifacts
themselves, not our beliefs about them.

Checks
  1. LINE ENDINGS   every shell script is LF-only (CRLF is fatal on the cluster)
  2. SHELL SYNTAX   every shell script passes `bash -n`
  3. PYTHON SYNTAX  every python file parses
  4. PYTHON IMPORT  every importable module imports without side effects
  5. TESTS          the scorer test suites pass
  6. DOC NUMBERS    numbers quoted in the findings/evidence docs are recomputed
                    from the result JSONs and must match exactly
  7. CLUSTER PARITY (optional, --cluster) files pushed to Gilbreth are
                    byte-identical to the local copies, by sha256

Exit code 0 only if every check passes.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

FAILURES: List[str] = []
CHECKS = 0


def ok(msg: str) -> None:
    global CHECKS
    CHECKS += 1
    print(f"  [ok]   {msg}")


def fail(msg: str) -> None:
    global CHECKS
    CHECKS += 1
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def sh_files() -> List[str]:
    out = []
    for root, _dirs, files in os.walk(HERE):
        if ".git" in root:
            continue
        for f in files:
            if f.endswith((".sh", ".sbatch")):
                out.append(os.path.join(root, f))
    return sorted(out)


def py_files() -> List[str]:
    out = []
    for root, _dirs, files in os.walk(HERE):
        if ".git" in root or "__pycache__" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                out.append(os.path.join(root, f))
    for root, _dirs, files in os.walk(os.path.join(REPO, "research", "tools")):
        for f in files:
            if f.endswith(".py"):
                out.append(os.path.join(root, f))
    return sorted(out)


def check_line_endings() -> None:
    print("\n[1] line endings (CRLF is fatal for cluster shell scripts)")
    for p in sh_files():
        data = open(p, "rb").read()
        if b"\r" in data:
            fail(f"{os.path.relpath(p, REPO)} contains CR bytes "
                 f"({data.count(chr(13).encode())} found)")
        else:
            ok(f"{os.path.relpath(p, REPO)} LF-only")


_MOUNT_PREFIX: str | None = None


def _detect_mount_prefix() -> str:
    """Discover how this bash maps Windows drives, by probing — not guessing.

    `bash -n C:\\Users\\...` eats the backslashes and reports 'No such file'
    for every script, which masquerades as dozens of syntax failures. The
    POSIX form differs by installation: many Git Bash setups use /c/..., this
    one is configured with /mnt/c/.... Probe a file we know exists and cache
    whichever prefix resolves, so the check works on either.
    """
    global _MOUNT_PREFIX
    if _MOUNT_PREFIX is not None:
        return _MOUNT_PREFIX
    probe = os.path.abspath(__file__).replace("\\", "/")
    drive, rest = probe[0].lower(), probe[2:]
    for prefix in ("/mnt/", "/"):
        cand = f"{prefix}{drive}{rest}"
        r = subprocess.run(["bash", "-c", f"test -f '{cand}'"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            _MOUNT_PREFIX = prefix
            return prefix
    _MOUNT_PREFIX = "/"          # fall back; failures will be reported honestly
    return _MOUNT_PREFIX


def _posix(path: str) -> str:
    p = os.path.abspath(path).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = f"{_detect_mount_prefix()}{p[0].lower()}{p[2:]}"
    return p


def check_shell_syntax() -> None:
    print("\n[2] shell syntax (bash -n)")
    bash = "bash"
    for p in sh_files():
        try:
            r = subprocess.run([bash, "-n", _posix(p)], capture_output=True,
                               text=True, timeout=60)
        except Exception as exc:  # noqa: BLE001
            fail(f"{os.path.relpath(p, REPO)} could not run bash -n: {exc}")
            continue
        if r.returncode == 0:
            ok(f"{os.path.relpath(p, REPO)}")
        else:
            fail(f"{os.path.relpath(p, REPO)}: {r.stderr.strip().splitlines()[:2]}")


def check_python_syntax() -> None:
    print("\n[3] python syntax")
    for p in py_files():
        try:
            ast.parse(open(p, encoding="utf-8").read())
            ok(os.path.relpath(p, REPO))
        except SyntaxError as exc:
            fail(f"{os.path.relpath(p, REPO)}: {exc}")
        except UnicodeDecodeError as exc:
            fail(f"{os.path.relpath(p, REPO)}: not valid UTF-8: {exc}")


def check_imports() -> None:
    print("\n[4] python import (no import-time side effects / missing names)")
    mods = ["decoupled_scorer", "bridge_predictions", "dominance_guard",
            "nonidentifiability_sweep", "nonidentifiability_empirical",
            "nonidentifiability_thresholds", "paired_analysis"]
    for m in mods:
        path = os.path.join(HERE, m + ".py")
        if not os.path.exists(path):
            fail(f"{m}.py missing")
            continue
        r = subprocess.run([sys.executable, "-c",
                            f"import sys; sys.path.insert(0, r'{HERE}'); import {m}"],
                           capture_output=True, text=True, timeout=180)
        if r.returncode == 0:
            ok(f"import {m}")
        else:
            fail(f"import {m}: {r.stderr.strip().splitlines()[-1:]}")


def check_tests() -> None:
    print("\n[5] test suites")
    tests = [t for t in ("test_decoupled_scorer.py",
                         "test_mass_conservation_real.py")
             if os.path.exists(os.path.join(HERE, t))]
    if not tests:
        fail("no test files found")
        return
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests],
                       capture_output=True, text=True, cwd=HERE, timeout=900)
    tail = (r.stdout or r.stderr).strip().splitlines()[-1:] or ["(no output)"]
    if r.returncode == 0:
        ok(f"pytest {' '.join(tests)} -> {tail[0]}")
    else:
        fail(f"pytest failed -> {tail[0]}")


def check_bridge() -> None:
    """The analytic bridge underwrites a paper claim; keep it in the gate.

    Proposition 4 (the admissible-HM band) is the device that lets a
    single-corpus letter stand, and bridge_checks.py also pins the honest
    negatives — the macro inequality is refuted as a theorem, and eps=0 is not
    collapse-free. If any of that silently changes, the paper's framing breaks.
    """
    print("\n[6a] analytic bridge (bridge_checks.py)")
    p = os.path.join(HERE, "bridge_checks.py")
    if not os.path.exists(p):
        fail("bridge_checks.py missing")
        return
    r = subprocess.run([sys.executable, p], capture_output=True, text=True,
                       cwd=HERE, timeout=1800)
    if r.returncode == 0 and "ALL BRIDGE CHECKS PASSED" in (r.stdout or ""):
        n = (r.stdout or "").count("[PASS]")
        ok(f"bridge_checks.py: {n} checks passed")
    else:
        tail = (r.stdout or r.stderr).strip().splitlines()[-2:]
        fail(f"bridge_checks.py failed: {tail}")


def _load(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def check_doc_numbers() -> None:
    """Recompute every headline number we quote, from the artifacts."""
    print("\n[6] documented numbers vs result artifacts")
    gil = os.path.join(HERE, "results", "gilbreth")

    # 6a. headline HM / Loc-F for both systems
    expect = {
        "A_polytune_maestro_strict_eps05.json": (0.2931, 0.6494),
        "B_laddersym_maestro_unprompted_strict_eps05.json": (0.1994, 0.7132),
        "B_laddersym_maestro_prompted_strict_eps05.json": (0.1754, 0.7588),
    }
    for fn, (hm_x, loc_x) in expect.items():
        p = os.path.join(gil, fn)
        if not os.path.exists(p):
            fail(f"{fn} missing")
            continue
        t = _load(p)["decoupled"]["per_tau"][0]
        hm, loc = round(t["hm"], 4), round(t["localization"]["f1"], 4)
        if (hm, loc) == (hm_x, loc_x):
            ok(f"{fn}: HM@50={hm} Loc-F={loc}")
        else:
            fail(f"{fn}: documented HM={hm_x} Loc-F={loc_x}, "
                 f"artifact HM={hm} Loc-F={loc}")

    # 6b. HM recomputed from the confusion matrix itself, not read back
    for fn in expect:
        p = os.path.join(gil, fn)
        if not os.path.exists(p):
            continue
        t = _load(p)["decoupled"]["per_tau"][0]
        conf = t["confusion"]
        types = list(conf)
        off = sum(conf[r][c] for r in types for c in types if r != c)
        tot = sum(conf[r][c] for r in types for c in types)
        recomputed = off / tot if tot else None
        if recomputed is not None and abs(recomputed - t["hm"]) < 1e-12:
            ok(f"{fn}: HM recomputed from confusion == stored ({recomputed:.6f})")
        else:
            fail(f"{fn}: HM recompute {recomputed} != stored {t['hm']}")

    # 6c. paired analysis figures quoted in the summary
    p = os.path.join(gil, "paired_analysis.json")
    if os.path.exists(p):
        d = _load(p)
        checks = [("hm", "n_pieces_favouring_a", 170),
                  ("loc_f1", "n_pieces_favouring_b", 173)]
        for metric, key, want in checks:
            got = d[metric][key]
            (ok if got == want else fail)(
                f"paired_analysis {metric}.{key} = {got} (documented {want})")
    else:
        fail("paired_analysis.json missing")

    # 6c-ii. prompted-vs-unprompted ablation (per-piece paired)
    p = os.path.join(gil, "paired_prompted_vs_unprompted.json")
    if os.path.exists(p):
        d = _load(p)
        # DIRECTION MATTERS. system_a = prompted. HM: LOWER is better, so
        # "pieces where prompted wins" = n_pieces_favouring_b (prompted holds
        # the smaller value). Loc-F: HIGHER is better -> n_pieces_favouring_a.
        # Reading the wrong field here is the exact lower-is-better sign trap
        # that the gate exists to catch.
        hm_fav = d["hm"]["n_pieces_favouring_b"]      # prompted better = lower HM
        lf_fav = d["loc_f1"]["n_pieces_favouring_a"]  # prompted better = higher F
        if d["n_used"] == 177 and d["n_excluded_undefined_hm"] == 0:
            ok(f"ablation: paired over all 177 pieces, none dropped")
        else:
            fail(f"ablation: n_used={d['n_used']} excluded={d['n_excluded_undefined_hm']}")
        # HM small effect, Loc-F medium effect — the decomposition IS the finding
        if hm_fav == 125 and lf_fav == 163:
            ok(f"ablation: prompted favoured on HM {hm_fav}/177, Loc-F {lf_fav}/177")
        else:
            fail(f"ablation: HM fav {hm_fav} (want 125), Loc-F fav {lf_fav} (want 163)")
        hm_d, lf_d = d["hm"]["cliffs_delta"], d["loc_f1"]["cliffs_delta"]
        if abs(hm_d) < abs(lf_d):
            ok(f"ablation: |Cliff HM|={abs(hm_d):.3f} < |Cliff Loc-F|={abs(lf_d):.3f} "
               f"(localization gain exceeds classification gain — the claim)")
        else:
            fail(f"ablation: effect-size ordering wrong (HM {hm_d}, Loc-F {lf_d})")
    else:
        fail("paired_prompted_vs_unprompted.json missing")

    # 6c-iii. 3-system table: identical piece sets, all HM defined
    p = os.path.join(gil, "nonidentifiability_empirical_3sys.json")
    if os.path.exists(p):
        pp = _load(p)["per_piece"]
        sets = {k: set(r["piece"] for r in v) for k, v in pp.items()}
        base = next(iter(sets.values()))
        if len(pp) == 3 and all(s == base for s in sets.values()) and len(base) == 177:
            ok("3-system table: identical 177-piece set across polytune/unprompted/prompted")
        else:
            fail(f"3-system table: piece sets differ or wrong count "
                 f"({[len(s) for s in sets.values()]})")
    else:
        fail("nonidentifiability_empirical_3sys.json missing")

    # 6c-iv. pooled != per-piece-mean is EXPECTED (weighting), same direction.
    # Guard the honesty caveat: pooled HM must exceed per-piece-mean HM for
    # BOTH laddersym configs, and both must rank prompted below unprompted.
    try:
        pa = _load(os.path.join(gil, "paired_prompted_vs_unprompted.json"))
        pooled = {
            "prompted": _load(os.path.join(
                gil, "B_laddersym_maestro_prompted_strict_eps05.json"
            ))["decoupled"]["per_tau"][0]["hm"],
            "unprompted": _load(os.path.join(
                gil, "B_laddersym_maestro_unprompted_strict_eps05.json"
            ))["decoupled"]["per_tau"][0]["hm"],
        }
        ppm = {"prompted": pa["hm"]["mean_a"], "unprompted": pa["hm"]["mean_b"]}
        same_dir = (pooled["prompted"] < pooled["unprompted"]) and \
                   (ppm["prompted"] < ppm["unprompted"])
        differ = abs(pooled["prompted"] - ppm["prompted"]) > 1e-6
        if same_dir and differ:
            ok("pooled vs per-piece-mean HM: differ in value, agree in direction "
               "(caveat is real and documented)")
        else:
            fail(f"pooled/per-piece HM caveat broken: pooled={pooled} ppm={ppm}")
    except (KeyError, TypeError) as exc:
        fail(f"pooled/per-piece caveat check errored: {exc}")

    # 6d. constructed sweep really does hold shipped stats invariant
    p = os.path.join(HERE, "results", "nonidentifiability_sweep.json")
    if os.path.exists(p):
        s = _load(p)["summary"]
        (ok if s["shipped_invariant"] else fail)(
            f"sweep: shipped_invariant={s['shipped_invariant']}, "
            f"HM span {s['hm_span']:.4f}")
    else:
        fail("nonidentifiability_sweep.json missing")

    # 6f. letter Sec. V sensitivity claim: the matches gained between tau=50
    # and tau=500 ms are "predominantly off-diagonal (78.7%--83.1% across
    # configurations)". Re-derive each percentage from the incremental
    # confusion mass (per_tau[500] - per_tau[50]) and require the printed
    # 78.7--83.1 range to bracket every value at one-decimal rounding.
    for fn, want in [
            ("A_polytune_maestro_strict_eps05.json", 82.1),
            ("B_laddersym_maestro_unprompted_strict_eps05.json", 83.1),
            ("B_laddersym_maestro_prompted_strict_eps05.json", 78.7)]:
        p = os.path.join(gil, fn)
        if not os.path.exists(p):
            fail(f"{fn} missing (incremental off-diagonal check)")
            continue
        per_tau = {t["tau_ms"]: t for t in _load(p)["decoupled"]["per_tau"]}
        c50, c500 = per_tau[50]["confusion"], per_tau[500]["confusion"]
        types = list(c50)
        inc = {(r, c): c500[r][c] - c50[r][c] for r in types for c in types}
        tot = sum(inc.values())
        off = sum(v for (r, c), v in inc.items() if r != c)
        pct = 100.0 * off / tot if tot else float("nan")
        if abs(pct - want) <= 0.055 and 78.7 <= round(pct, 1) <= 83.1:
            ok(f"{fn}: incremental off-diagonal 50->500 ms = {off}/{tot} "
               f"({pct:.1f}%), inside printed 78.7--83.1%")
        else:
            fail(f"{fn}: incremental off-diagonal {off}/{tot} ({pct:.1f}%) "
                 f"vs documented {want}% / printed range 78.7--83.1%")

    # 6g. letter Sec. V replication paragraph: nine numbers. Pooled
    # missed/extra F1 from the shipped artifacts, published-style per-piece
    # (macro) F1 from replication_macro.json, and the "within 0.034 of every
    # published value" bound on the macro-vs-published gaps.
    published = {"polytune": (0.268, 0.720),          # chou2025detecting
                 "laddersym_prompted": (0.563, 0.864)}  # chou2025multimodal
    letter_pooled = {"polytune": (0.261, 0.683),
                     "laddersym_prompted": (0.485, 0.828)}
    letter_macro = {"polytune": (0.287, 0.713),
                    "laddersym_prompted": (0.530, 0.854)}
    stem = {"polytune": "A_polytune_maestro",
            "laddersym_prompted": "B_laddersym_maestro_prompted"}
    macro_p = os.path.join(gil, "replication_macro.json")
    if not os.path.exists(macro_p):
        fail("replication_macro.json missing (replication-paragraph check)")
    else:
        macro = _load(macro_p)["systems"]
        gaps = []
        for sysname in ("polytune", "laddersym_prompted"):
            sp = os.path.join(gil, f"{stem[sysname]}_shipped.json")
            if not os.path.exists(sp):
                fail(f"{stem[sysname]}_shipped.json missing")
                continue
            s50 = _load(sp)["shipped_50ms"]
            pooled = (round(s50["missed"]["f1"], 3), round(s50["extra"]["f1"], 3))
            if pooled == letter_pooled[sysname]:
                ok(f"replication pooled {sysname}: {pooled[0]:.3f}/{pooled[1]:.3f} "
                   f"as printed")
            else:
                fail(f"replication pooled {sysname}: artifact {pooled} != "
                     f"printed {letter_pooled[sysname]}")
            if sysname == "polytune":
                # letter Sec. V dominant-cell clause: "the low missed precision
                # under the published recipe (0.228 for Polytune)" —
                # TP/(TP+FP) of shipped_50ms (artifact key name unchanged)
                prec = round(s50["missed"]["precision"], 3)
                tp, fp = s50["missed"]["tp"], s50["missed"]["fp"]
                if prec == 0.228 and round(tp / (tp + fp), 3) == 0.228:
                    ok(f"letter dominant-cell clause: Polytune shipped missed "
                       f"precision {tp}/{tp + fp} = {prec:.3f} == printed 0.228")
                else:
                    fail(f"Polytune shipped missed precision {prec} "
                         f"({tp}/{tp + fp}) != printed 0.228")
            m = macro[sysname]
            mac = (round(m["macro_f1_missed"], 3), round(m["macro_f1_extra"], 3))
            if mac == letter_macro[sysname]:
                ok(f"replication macro {sysname}: {mac[0]:.3f}/{mac[1]:.3f} "
                   f"as printed")
            else:
                fail(f"replication macro {sysname}: artifact {mac} != "
                     f"printed {letter_macro[sysname]}")
            # cross-check: macro JSON's pooled F1 agrees with the shipped JSON
            jp = (round(m["pooled_f1_missed"], 3), round(m["pooled_f1_extra"], 3))
            if jp != pooled:
                fail(f"replication {sysname}: replication_macro pooled {jp} "
                     f"disagrees with shipped artifact {pooled}")
            gaps += [abs(m["macro_f1_missed"] - published[sysname][0]),
                     abs(m["macro_f1_extra"] - published[sysname][1])]
        if gaps:
            if max(gaps) <= 0.034:
                ok(f"replication: max macro-vs-published gap "
                   f"{max(gaps):.4f} <= printed bound 0.034")
            else:
                fail(f"replication: max macro-vs-published gap {max(gaps):.4f} "
                     f"exceeds printed bound 0.034")
        # letter Sec. V replication clause: "Polytune missed overshooting by
        # 0.019" — macro missed F1 minus the published 0.268, and it must be
        # an overshoot (positive), not an undershoot.
        if "polytune" in macro:
            over = macro["polytune"]["macro_f1_missed"] - published["polytune"][0]
            if over > 0 and round(over, 3) == 0.019:
                ok(f"replication: Polytune missed macro overshoot "
                   f"{over:.4f} rounds to printed 0.019")
            else:
                fail(f"replication: Polytune missed macro overshoot {over:.4f} "
                     f"!= printed 0.019")

    # 6e. coco census percentages quoted in F6
    for fn, want_bad, want_tot in [
            ("C_polytune_coco_track_census.json", 2285, 4284),
            ("D_laddersym_coco_prompted_track_census.json", 2192, 4284),
            ("D_laddersym_coco_unprompted_track_census.json", 2170, 4284)]:
        p = os.path.join(gil, fn)
        if not os.path.exists(p):
            print(f"  [skip] {fn} not synced locally")
            continue
        c = _load(p)
        h = {int(k): v for k, v in c["histogram"].items()}
        bad = sum(v for k, v in h.items() if k != 3)
        if bad == want_bad and c["n_files"] == want_tot:
            ok(f"{fn}: under-specified {bad}/{want_tot}")
        else:
            fail(f"{fn}: documented {want_bad}/{want_tot}, "
                 f"artifact {bad}/{c['n_files']}")


def check_rescore_v110() -> None:
    """The v1.1.0 rescore must agree with the shipped numbers, and its
    mass-conservation guard must actually hold.

    Through v1.0.0 `mass_conserved()` was tautological (verifier finding R1),
    so the shipped artifacts were never genuinely consistency-checked. Job
    11341983 rescored the same predictions under v1.1.0, where the guard is
    real. HM/Loc-F must be bit-identical (neither reads `spurious`, the only
    computational change); any drift here is a real defect, not a rounding
    artifact.
    """
    print("\n[6b] v1.1.0 rescore parity + real mass-conservation guard")
    v110 = os.path.join(HERE, "results", "gilbreth_v110")
    if not os.path.isdir(v110):
        fail("results/gilbreth_v110 missing (run gilbreth_rescore_v110.sbatch)")
        return

    shipped_dir = os.path.join(HERE, "results", "gilbreth")
    n_pairs = 0
    for fn in sorted(os.listdir(v110)):
        if not fn.endswith(".json"):
            continue
        new = _load(os.path.join(v110, fn))
        old_p = os.path.join(shipped_dir, fn)
        if not os.path.exists(old_p) or new is None:
            continue
        old = _load(old_p)
        if old is None or "bootstrap" not in new or "50" not in new.get("bootstrap", {}):
            continue
        for key in ("point_hm", "point_loc_f1"):
            a, b = old["bootstrap"]["50"][key], new["bootstrap"]["50"][key]
            n_pairs += 1
            if a is None and b is None:
                continue
            if a is None or b is None or abs(a - b) > 1e-12:
                fail(f"{fn}: {key} {a} -> {b} DIFFERS under v1.1.0")
        if new["config"].get("scorer_version") != "1.1.0":
            fail(f"{fn}: rescore artifact is not v1.1.0")
    if n_pairs:
        ok(f"v1.1.0 rescore: {n_pairs} HM/Loc-F values bit-identical to shipped")

    total = conserved = 0
    for fn in sorted(os.listdir(v110)):
        if not fn.endswith(".json"):
            continue
        d = _load(os.path.join(v110, fn))
        for pt in (d or {}).get("decoupled", {}).get("per_tau", []):
            total += 1
            conserved += bool(pt.get("mass_conserved"))
    if total and conserved == total:
        ok(f"mass_conserved (real guard): {conserved}/{total} tau points")
    elif total:
        fail(f"mass_conserved: only {conserved}/{total} tau points hold")
    else:
        fail("no per_tau entries found in rescore artifacts")


def check_figs_tables() -> None:
    """Figures/tables are generated from artifacts; prove they still match.

    Two failure modes this catches: (a) a committed tab_*.tex was hand-edited or
    left stale after the data changed -- we regenerate to a temp dir and byte-diff;
    (b) the figure PDF drifted to Type-3 fonts, which IEEE PDF eXpress rejects.
    """
    print("\n[8] figures/tables integrity")
    figs = os.path.join(HERE, "figs")
    if not os.path.isdir(figs):
        fail("figs/ directory missing")
        return

    # (a) regenerate tables into a temp dir and compare to committed copies
    import tempfile
    tables = ("tab_main.tex", "tab_ablation.tex", "tab_coco.tex",
              "tab_null.tex", "tab_confusion.tex")
    committed_missing = [t for t in tables if not os.path.exists(os.path.join(figs, t))]
    if committed_missing:
        fail(f"committed tables missing: {committed_missing}")
    else:
        with tempfile.TemporaryDirectory() as td:
            env = dict(os.environ, MER_TABLE_OUTDIR=td)
            r = subprocess.run([sys.executable, "make_tables.py"],
                               cwd=figs, env=env, capture_output=True, text=True)
            if r.returncode != 0:
                fail(f"make_tables.py failed: {r.stderr.strip()[:200]}")
            else:
                for t in tables:
                    regen = os.path.join(td, t)
                    comm = os.path.join(figs, t)
                    if not os.path.exists(regen):
                        fail(f"{t}: not regenerated")
                        continue
                    a = open(regen, encoding="utf-8").read()
                    b = open(comm, encoding="utf-8").read()
                    if a == b:
                        ok(f"{t}: committed == regenerated-from-artifacts")
                    else:
                        fail(f"{t}: committed table DIFFERS from artifacts (stale/edited)")

    # (b) figure PDFs present and free of Type-3 fonts
    for pdf_name in ("fig1_two_scenario.pdf", "fig2_measured.pdf",
                     "figS_witness.pdf"):
        pdf = os.path.join(figs, pdf_name)
        if not os.path.exists(pdf):
            fail(f"{pdf_name} missing (run its generator)")
        elif b"/Type3" in open(pdf, "rb").read():
            fail(f"{pdf_name} contains Type-3 fonts (IEEE PDF eXpress rejects)")
        else:
            ok(f"{pdf_name}: no Type-3 fonts (IEEE-compliant)")

    # (c) Figure 1's exact claim: the two real-data scenarios have IDENTICAL
    # shipped reports but different HM. Re-run the verifier, require it holds.
    r = subprocess.run([sys.executable, "verify_two_scenario.py"],
                       cwd=figs, capture_output=True, text=True)
    if r.returncode == 0 and "CLAIM HOLDS: True" in r.stdout:
        ok("Fig.1 two-scenario claim: identical shipped, HM 1 vs 0 (exact)")
    else:
        fail(f"Fig.1 two-scenario claim FAILED to verify: {r.stdout.strip()[-160:]}")

    # (d) null-model robustness: matched co-locations exceed the shifted null
    # (HM is real signal, not dense-passage coincidence).
    for cfg in ("A_polytune_maestro", "B_laddersym_maestro_unprompted",
                "B_laddersym_maestro_prompted"):
        p = os.path.join(HERE, "results", "gilbreth", f"null_colo_{cfg}.json")
        if not os.path.exists(p):
            fail(f"null_colo_{cfg}.json missing")
            continue
        d = _load(p)
        et = d.get("enrichment_total", 0) or 0
        pt = d.get("p_total_ge_observed", 1)
        if et >= 2.0 and pt <= 0.01:
            ok(f"null model {cfg[:14]}: matched-total {et:.1f}x, p={pt:.3f}")
        else:
            fail(f"null model {cfg}: enrichment {et:.1f}x p={pt} "
                 f"(need >=2x, <=0.01)")


def check_cluster_parity() -> None:
    print("\n[7] cluster parity (sha256 local vs Gilbreth)")
    pairs = [
        ("decoupled_scorer.py", "repo/experiments/decoupled_scorer.py"),
        ("bridge_predictions.py", "repo/experiments/bridge_predictions.py"),
        ("setup/gilbreth_run_stage.sh", "repo/experiments/setup/gilbreth_run_stage.sh"),
        ("setup/gilbreth_stage.sbatch", "repo/experiments/setup/gilbreth_stage.sbatch"),
    ]
    ssh = ["ssh", "-i", os.path.expanduser("~/.ssh/id_ed25519_anvil"),
           "-o", "BatchMode=yes", "-o", "ConnectTimeout=25",
           "dcharapa@gilbreth.rcac.purdue.edu"]
    for local_rel, remote_rel in pairs:
        lp = os.path.join(HERE, local_rel)
        if not os.path.exists(lp):
            fail(f"local {local_rel} missing")
            continue
        lh = hashlib.sha256(open(lp, "rb").read()).hexdigest()
        remote = f"/scratch/gilbreth/dcharapa/mer/{remote_rel}"
        r = subprocess.run(ssh + [f"sha256sum {remote} 2>/dev/null | cut -d' ' -f1"],
                           capture_output=True, text=True, timeout=120)
        rh = (r.stdout or "").strip().splitlines()
        rh = rh[-1] if rh else ""
        if rh == lh:
            ok(f"{local_rel} identical on cluster")
        elif not rh:
            fail(f"{local_rel}: could not read remote hash")
        else:
            fail(f"{local_rel}: DIFFERS (local {lh[:12]}, remote {rh[:12]})")


def main() -> int:
    do_cluster = "--cluster" in sys.argv
    print("verify_shipped — mechanical re-verification of shipped artifacts")
    check_line_endings()
    check_shell_syntax()
    check_python_syntax()
    check_imports()
    check_tests()
    check_bridge()
    check_doc_numbers()
    check_rescore_v110()
    check_figs_tables()
    if do_cluster:
        check_cluster_parity()
    else:
        print("\n[7] cluster parity SKIPPED (pass --cluster to enable)")

    print(f"\n{'=' * 62}")
    if FAILURES:
        print(f"FAIL — {len(FAILURES)} of {CHECKS} checks failed:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
