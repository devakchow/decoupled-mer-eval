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
import math
import os
import re
import shutil
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


_BASH: str | None = None
_BASH_RESOLVED = False


def _find_bash() -> str | None:
    """Resolve a WORKING bash once, by probing — not by trusting PATH.

    On some Windows setups PATH resolves to the broken WSL stub
    C:\\Windows\\System32\\bash.exe, which exits nonzero and emits UTF-16
    garbage ("Catastrophic failure") for every invocation — masquerading as
    dozens of syntax failures. Probe each candidate with `bash -c true`; a
    working bash returns 0 with no NUL bytes in its output. Cache the first
    winner; None means no working bash on this machine.
    """
    global _BASH, _BASH_RESOLVED
    if _BASH_RESOLVED:
        return _BASH
    _BASH_RESOLVED = True
    candidates = [shutil.which("bash"),
                  r"C:\Program Files\Git\bin\bash.exe",
                  r"C:\Program Files\Git\usr\bin\bash.exe"]
    for cand in candidates:
        if not cand or not os.path.exists(cand):
            continue
        try:
            r = subprocess.run([cand, "-c", "true"], capture_output=True,
                               timeout=60)
        except Exception:  # noqa: BLE001
            continue
        if r.returncode == 0 and b"\x00" not in (r.stdout + r.stderr):
            _BASH = cand
            return _BASH
    return None


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
    bash = _find_bash() or "bash"
    for prefix in ("/mnt/", "/"):
        cand = f"{prefix}{drive}{rest}"
        r = subprocess.run([bash, "-c", f"test -f '{cand}'"],
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
    bash = _find_bash()
    files = sh_files()
    if bash is None:
        print(f"  [warn] SKIPPED {len(files)} shell-syntax checks: "
              f"no working bash on this machine")
        return
    for p in files:
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

    # 6h. supplementary CocoChorales bias clause: "85.6% of Polytune's
    # three-track stems correspond to references containing both error
    # classes, versus 43.2% in the population". Re-derived from the evidence
    # artifacts: 1711/1999 (VERIFY_joingt.json, npred=3 rows) and 1852/4284
    # (VERIFY_gtcensus.json, present pattern (1,1,1)).
    jp = os.path.join(HERE, "evidence", "VERIFY_joingt.json")
    gp = os.path.join(HERE, "evidence", "VERIFY_gtcensus.json")
    if not os.path.exists(jp):
        fail("evidence/VERIFY_joingt.json missing (supp census check)")
    else:
        joint = _load(jp)["C_polytune_coco"]["joint"]
        three = {k: v for k, v in joint.items() if k.startswith("npred=3 ")}
        tot3 = sum(three.values())
        both3 = three.get("npred=3 gt=(1, 1, 1)", 0)
        pct3 = 100.0 * both3 / tot3 if tot3 else float("nan")
        if (both3, tot3) == (1711, 1999) and round(pct3, 1) == 85.6:
            ok(f"supp census: Polytune 3-trk stems with both error classes "
               f"{both3}/{tot3} ({pct3:.1f}%) == printed 85.6%")
        else:
            fail(f"supp census: 3-trk both-classes {both3}/{tot3} "
                 f"({pct3:.1f}%) != printed 85.6% (1711/1999)")
    if not os.path.exists(gp):
        fail("evidence/VERIFY_gtcensus.json missing (supp census check)")
    else:
        cen = _load(gp)
        n_all = cen["n_stems"]
        both_pop = cen["present_pattern_counts"]["(1, 1, 1)"]
        pctp = 100.0 * both_pop / n_all if n_all else float("nan")
        if (both_pop, n_all) == (1852, 4284) and round(pctp, 1) == 43.2:
            ok(f"supp census: population stems with both error classes "
               f"{both_pop}/{n_all} ({pctp:.1f}%) == printed 43.2%")
        else:
            fail(f"supp census: population both-classes {both_pop}/{n_all} "
                 f"({pctp:.1f}%) != printed 43.2% (1852/4284)")

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

    # 6i. garden-path guard. "the onset ties routine in quantized MIDI" reads
    # as a garden path ("ties routine" parses as a noun phrase). The defect has
    # been reintroduced three times, so the guard pins the invariant -- some
    # disambiguating word before "in quantized MIDI" -- rather than one exact
    # phrasing, which went stale when v5 chose "common" over "that are routine".
    if not os.path.isdir(os.path.join(REPO, "proposal")):
        print("  [skip] garden-path guard: proposal/ not in this distribution")
        return
    for rel in (os.path.join("proposal", "spl_letter_v5.tex"),
                os.path.join("proposal", "arxiv-package", "spl_letter.tex")):
        p = os.path.join(REPO, rel)
        if not os.path.exists(p):
            fail(f"{rel} missing (garden-path guard)")
            continue
        with open(p, encoding="utf-8") as fh:
            txt = fh.read().replace("\n", " ")
        good = ("the onset ties that are routine" in txt
                or "the onset ties common in quantized" in txt)
        if good and "onset ties routine in quantized" not in txt:
            ok(f"{rel}: onset-ties wording disambiguated, "
               f"garden-path form absent")
        else:
            fail(f"{rel}: garden-path regression -- Sec. III must not read "
                 f"'onset ties routine in quantized MIDI' bare; use 'that are "
                 f"routine' or 'common'")


def _prose(name: str):
    """Whitespace-normalized text of proposal/<name>, or None (with one [skip]
    line) in the public code release, which ships without the proposal/ tree;
    artifact-only assertions in the caller still run."""
    path = os.path.join(REPO, "proposal", name)
    if not os.path.isdir(os.path.join(REPO, "proposal")):
        print("  [skip] prose pins: proposal/ tree not in this distribution")
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return re.sub(r"\s+", " ", fh.read())


def check_score_filter() -> None:
    """[9e] Score-consistency filter: every printed number in the letter's
    'The diagnosis is actionable' paragraph and its conclusion clause derives
    from results/{gilbreth,gilbreth_ei}/*_scorefilter50.json against the
    unfiltered shipped/strict artifacts; the paired per-piece contrast comes
    from results/cluster/score_filter_paired_ci.json."""
    print("\n[9e] score-consistency filter (letter Sec. IV paragraph + conclusion)")
    tex = _prose("spl_letter_v5.tex")

    def pin(label, printed):
        if tex is None:
            return
        if printed in tex:
            ok(f"{label}: '{printed}'")
        else:
            fail(f"{label}: derived '{printed}' NOT in letter")

    E = ("A_polytune_maestro", "B_laddersym_maestro_unprompted", "B_laddersym_maestro_prompted")
    EI = ("A_polytune_maestro_ei", "B_laddersym_maestro_ei_unprompted", "B_laddersym_maestro_ei_prompted")

    def rows(stems, sub):
        out = []
        for st in stems:
            d = os.path.join(HERE, "results", sub)
            f = _load(os.path.join(d, st + "_scorefilter50.json"))
            bs = _load(os.path.join(d, st + "_shipped.json"))["shipped_50ms"]
            bt = _load(os.path.join(d, st + "_strict_eps05.json"))
            t50 = lambda x: [r for r in x["decoupled"]["per_tau"] if r["tau_ms"] == 50][0]
            out.append(dict(sf=f["score_filter"], fs=f["shipped_50ms"], bs=bs,
                            rf=t50(f), rb=t50(bt), cf=f["bootstrap"]["50"],
                            cb=bt["bootstrap"]["50"]))
        return out

    e, ei = rows(E, "gilbreth"), rows(EI, "gilbreth_ei")
    for r in e + ei:
        if r["sf"]["anchor_s"] != 0.05 or r["sf"]["n_dropped"] != r["sf"]["n_missed_before"] - r["sf"]["n_missed_after"]:
            fail("score-filter artifact inconsistent (anchor or counts)")
        if abs(r["fs"]["missed"]["recall"] - r["bs"]["missed"]["recall"]) >= 5e-5:
            fail("filter changed missed recall beyond four decimals")
        if r["fs"]["extra"] != r["bs"]["extra"]:
            fail("filter touched the extra class")
        if r["rf"]["localization"]["f1"] < r["rb"]["localization"]["f1"] - 1e-3:
            fail("filter lowered F by more than 0.001")
    ok("filter: recall unchanged to 4 decimals, extra class identical, F unchanged or higher (6 configs)")
    drop = lambda rs: [100 * r["sf"]["n_dropped"] / r["sf"]["n_missed_before"] for r in rs]
    pin("E drop shares", "removes $%d/%d/%d\\%%$ of the systems'" % tuple(round(x) for x in drop(e)))
    pin("E missed F1 before/after", "our replication, from $%.3f/%.3f/%.3f$ to $%.3f/%.3f/%.3f$"
        % (*[r["bs"]["missed"]["f1"] for r in e], *[r["fs"]["missed"]["f1"] for r in e]))
    pin("E mean error F1 before/after", "(mean error $F_1$ $%.3f/%.3f/%.3f$ to $%.3f/%.3f/%.3f$)"
        % (*[r["bs"]["_mean_error_f1"] for r in e], *[r["fs"]["_mean_error_f1"] for r in e]))
    if abs(e[1]["fs"]["_mean_error_f1"] - e[2]["bs"]["_mean_error_f1"]) >= 1e-3:
        fail("filtered unprompted mean error F1 no longer matches the prompted published mean within 0.001")
    else:
        ok("filtered unprompted mean error F1 = prompted published mean within 0.001")
    pin("E filtered HM", "lowering raw $\\mathrm{HM}$ to $%.3f/%.3f/%.3f$, below Table" % tuple(r["rf"]["hm"] for r in e))
    for r in e:
        if r["cf"]["hm_ci95"][1] >= r["cb"]["hm_ci95"][0]:
            fail("filtered HM interval not below the Table I interval")
    hm_f = [r["rf"]["hm"] for r in e]; f_f = [r["rf"]["localization"]["f1"] for r in e]
    if not (hm_f[0] > hm_f[1] > hm_f[2] and f_f[0] < f_f[1] < f_f[2]):
        fail("filtered ordering differs from Table I's")
    pin("EI drop shares", "removes $%d/%d/%d\\%%$, lifting" % tuple(round(x) for x in drop(ei)))
    pin("EI mean error F1 before/after", "from $%.3f/%.3f/%.3f$ to $%.3f/%.3f/%.3f$ and lowering"
        % (*[r["bs"]["_mean_error_f1"] for r in ei], *[r["fs"]["_mean_error_f1"] for r in ei]))
    pin("EI filtered HM", "lowering $\\mathrm{HM}$ to $%.3f/%.3f/%.3f$." % tuple(r["rf"]["hm"] for r in ei))
    gains = [r["fs"]["missed"]["f1"] - r["bs"]["missed"]["f1"] for r in e]
    pin("conclusion missed-F1 gain", "by $%.2f$ to $%.2f$" % (min(gains), max(gains)))
    pc = os.path.join(HERE, "results", "cluster", "score_filter_paired_ci.json")
    if os.path.exists(pc):
        P = _load(pc)
        for st in E + EI:
            st_ = P[st]["stats"]
            if not (st_["missed_f1"]["diff_ci95"][0] > 0 and st_["mean_error_f1"]["diff_ci95"][0] > 0
                    and st_["hm"]["diff_ci95"][1] < 0):
                fail(f"paired filter contrast does not exclude zero for {st}")
        ok("paired per-piece contrast excludes zero (missed F1, mean error F1 up; HM down) for all 6")
        for st in E:
            if P[st]["stats"]["loc_f"]["diff_ci95"][0] <= 0:
                fail(f"paired F difference does not exclude zero for {st}; the letter says it does")
    else:
        fail("results/cluster/score_filter_paired_ci.json missing")


def check_round17_supplement() -> None:
    """[9f] round-17 supplement additions: EI sweep endpoints, paired HM_G intervals, guard definition."""
    print("\n[9f] round-17 supplement pins")
    stex = _prose("spl_supplementary_v5.tex")
    def spin(label, printed):
        if stex is None: return
        if printed in stex: ok(f"{label}: '{printed}'")
        else: fail(f"{label}: derived '{printed}' NOT in supplement")
    gei = os.path.join(HERE, "results", "gilbreth_ei")
    ks = ["A_polytune_maestro_ei", "B_laddersym_maestro_ei_unprompted", "B_laddersym_maestro_ei_prompted"]
    per = lambda k, v: {x["tau_ms"]: x for x in _load(os.path.join(gei, f"{k}_{v}.json"))["decoupled"]["per_tau"]}
    spin("EI sweep 500 ms HM", "to $%.3f/%.3f/%.3f$ at $500$~ms" % tuple(per(k, "strict_eps05")[500]["hm"] for k in ks))
    spin("EI sweep 500 ms F", "to $%.3f/%.3f/%.3f$; the ordering holds" % tuple(per(k, "strict_eps05")[500]["localization"]["f1"] for k in ks))
    for v in ("strict_eps05", "strict_eps0", "pitchaware_eps05"):
        for t in (50, 75, 100, 150, 200, 500):
            h = [per(k, v)[t]["hm"] for k in ks]; f = [per(k, v)[t]["localization"]["f1"] for k in ks]
            if not (h[0] > h[1] > h[2] and f[0] < f[1] < f[2]):
                fail(f"EI ordering breaks at {v} tau={t}")
    ok("EI ordering holds at every tau under three variants")
    pc = _load(os.path.join(HERE, "results", "cluster", "paired_ci.json"))["paired_diff_ci95"]
    k1 = "A_polytune_maestro - B_laddersym_maestro_unprompted"; k2 = "B_laddersym_maestro_unprompted - B_laddersym_maestro_prompted"
    spin("paired HM_G intervals", "Polytune$-$unprompted [%.3f,\\,%.3f], unprompted$-$prompted [%.3f,\\,%.3f]"
         % (pc[k1]["hm_g"][0], pc[k1]["hm_g"][1], pc[k2]["hm_g"][0], pc[k2]["hm_g"][1]))
    if not (pc[k1]["hm_g"][0] > 0 and pc[k2]["hm_g"][0] > 0):
        fail("paired HM_G intervals no longer exclude zero")
    spin("dominance guard defined", "the dominance guard (each predicted track's class mapping diagonal-dominant against the reference)")
    spin("null coverage sentence", "in every test, the observed exceeds all 200 rotations.")


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


def check_letter_prose() -> None:
    """Re-derive every number the letter prints in prose (not in a generated
    table) from the artifacts, and assert the exact printed string is present.

    The generated tables are already gated in [8]; prose numbers were not, so a
    revision could silently leave a stale figure in the text while every table
    still verified. Each entry below states the artifact it comes from.
    """
    print("\n[9] letter prose numbers vs result artifacts")
    gil = os.path.join(HERE, "results", "gilbreth")
    # The public code release ships without the proposal/ tree; skip there.
    if not os.path.isdir(os.path.join(os.path.dirname(HERE), "proposal")):
        print("  [skip] proposal/ tree not in this distribution")
        return
    letter = os.path.join(os.path.dirname(HERE), "proposal", "spl_letter_v5.tex")
    if not os.path.exists(letter):
        fail("proposal/spl_letter_v5.tex missing")
        return
    with open(letter, "r", encoding="utf-8") as fh:
        tex = re.sub(r"\s+", " ", fh.read())

    stems = ("A_polytune_maestro", "B_laddersym_maestro_unprompted",
             "B_laddersym_maestro_prompted")

    def assert_in(label: str, printed: str) -> None:
        if printed in tex:
            ok(f"prose {label}: '{printed}' present and artifact-derived")
        else:
            fail(f"prose {label}: derived '{printed}' NOT found in letter")

    # 9a. null-model enrichment, mean- and max-based (results/gilbreth/null_colo_*)
    off_mean, tot_mean, off_max, tot_max = [], [], [], []
    for stem in stems:
        p_ = os.path.join(gil, f"null_colo_{stem}.json")
        if not os.path.exists(p_):
            fail(f"null_colo_{stem}.json missing")
            return
        d = _load(p_)
        off_mean.append(d["enrichment_off"])
        tot_mean.append(d["enrichment_total"])
        off_max.append(d["observed_off_diagonal"] / d["null_off_diagonal"]["max"])
        tot_max.append(d["observed_matched_total"] / d["null_matched_total"]["max"])
    assert_in("null total vs mean",
              f"${min(tot_mean):.1f}$--${max(tot_mean):.1f}\\times$")
    # 9b. TIDE three-way bins and the collapse census
    #     (results/cluster/collapse_validation.json, cells checksummed there)
    sys.path.insert(0, os.path.join(HERE, "figs"))
    import make_tables as MT  # noqa: E402

    hm_t, unf, pos, susp, mu_p, merge_pct, raw = [], [], [], [], [], [], []
    num_p = den_p = 0
    chg = []
    cv = _load(os.path.join(HERE, "results", "cluster",
                            "collapse_validation.json"))
    for short, stem in (("polytune", stems[0]),
                        ("laddersym_unprompted", stems[1]),
                        ("laddersym_prompted", stems[2])):
        b = MT.tide_bins(short)
        hm_t.append(b["hm"])
        chg.append(b["hm_charging_u"])
        if short == "polytune":
            num_p, den_p = b["numerator"], b["denominator"]
        unf.append(b["unfounded"])
        c = cv[stem]
        pos.append(c["wrong->wrong"]["in_score_removed"] / c["wrong->wrong"]["n"])
        susp.append(c["extra->wrong"]["in_score_removed"] / c["extra->wrong"]["n"])
        mu_p.append(c["_n_merges_pred"])
        t = [x for x in _load(os.path.join(gil, f"{stem}_shipped.json"))
             ["decoupled"]["per_tau"] if x["tau_ms"] == 50][0]
        cs = t["confusion_sparse"]
        off = sum(v for k, v in cs.items()
                  if k.split("->")[0] != k.split("->")[1])
        raw.append(off / t["n_localized"])
        merge_pct.append(c["extra->wrong"]["n"] / off)

    assert_in("HM after binning (Polytune, printed with counts)",
              "%d{,}%03d/%d{,}%03d=%.3f" % (num_p // 1000, num_p % 1000,
                                            den_p // 1000, den_p % 1000, hm_t[0]))
    assert_in("HM after binning (LadderSym)",
              "$%.3f$/$%.3f$" % (hm_t[1], hm_t[2]))
    assert_in("HM charging U", "gives $%.3f/%.3f/%.3f$" % tuple(chg))
    assert_in("span endpoints", "$[%.3f,\\,%.3f]$" % (hm_t[0], raw[0]))
    assert_in("unfounded share",
              "$%.1f\\%%$, $%.1f\\%%$, $%.1f\\%%$" % tuple(u * 100 for u in unf))
    assert_in("positive control",
              "$%.2f/%.2f/%.2f$" % tuple(pos))
    assert_in("suspect-cell genuine rate",
              "$%.3f$--$%.3f$" % (min(susp), max(susp)))
    assert_in("mu_p merges",
              "$\\mu_p=%s/%s/%s$" % tuple(f"{m:,}".replace(",", "{,}") for m in mu_p))
    # dominant-cell (extra->wrong) share of the raw hidden mass, printed as the
    # standard-rounding hull over the three configurations
    dom = []
    for st in ("A_polytune_maestro", "B_laddersym_maestro_unprompted",
               "B_laddersym_maestro_prompted"):
        d50 = [x for x in _load(os.path.join(gil, st + "_strict_eps05.json"))
               ["decoupled"]["per_tau"] if x["tau_ms"] == 50][0]
        off = sum(v for k, v in d50["confusion_sparse"].items()
                  if k.split("->")[0] != k.split("->")[1])
        dom.append(100 * d50["confusion_sparse"]["extra->wrong"] / off)
    assert_in("dominant-cell share hull",
              "cell carrying $%d$--$%d\\%%$ of the raw hidden mass"
              % (round(min(dom)), round(max(dom))))
    # 9b-2. replication agreement against the printed per-class F1 values.
    # LadderSym's abstract prints 0.563 for its missed note, but its own table
    # gives 0.563 as recall and 0.547 as F1; we compare against the F1.
    PRINTED_F1 = {"polytune": (0.268, 0.720),
                  "laddersym_prompted": (0.547, 0.864)}
    rep_m = _load(os.path.join(gil, "replication_macro.json"))
    dev = 0.0
    for k, (pm, pe) in PRINTED_F1.items():
        row = rep_m["systems"][k]
        dev = max(dev, abs(row["macro_f1_missed"] - pm),
                  abs(row["macro_f1_extra"] - pe))
    assert_in("replication agreement",
              "(within $%.3f$;" % (math.ceil(dev * 1000) / 1000))

    # 9c. supplement's reproducibility counts, derived from results/gilbreth_v110
    supp = os.path.join(os.path.dirname(HERE), "proposal",
                        "spl_supplementary_v5.tex")
    if os.path.exists(supp):
        with open(supp, "r", encoding="utf-8") as fh:
            stex = re.sub(r"\s+", " ", fh.read())
        v110 = os.path.join(HERE, "results", "gilbreth_v110")
        n_vals = n_tau = 0
        for fn in sorted(os.listdir(v110)):
            if not fn.endswith(".json"):
                continue
            d = _load(os.path.join(v110, fn))
            if "50" in (d or {}).get("bootstrap", {}):
                n_vals += 2  # point_hm, point_loc_f1
            n_tau += len((d or {}).get("decoupled", {}).get("per_tau", []))
        for label, printed in ((f"rescore value count",
                                f"bit-identically (${n_vals}$ $\\mathrm{{HM}}$/$F$ values)"),
                               (f"guard tau-point count",
                                f"holds at all ${n_tau}$ $(\\tau,\\text")):
            if printed in stex:
                ok(f"supplement {label}: '{printed}' artifact-derived")
            else:
                fail(f"supplement {label}: derived '{printed}' NOT in supplement")
    else:
        fail("proposal/spl_supplementary_v5.tex missing")

    # Synthetic-oracle recovery: planted HM* must be recovered exactly, the two
    # constructed systems must share a per-class F1 while differing in HM, and
    # the shipped per-class mean must be flat across the sweep (blind to HM).
    orc = _load(os.path.join(HERE, "oracle_report.json"))
    planted = orc["experiment_C"]["planted_hm"]
    rec = [r for r in orc["experiment_C"]["sweep"] if r["hm"] is not None]
    if rec and all(abs(r["hm"] - round(planted, 4)) < 5e-5 for r in rec):
        ok(f"oracle recovers planted HM*={planted:.4f} at all {len(rec)} tolerances")
    else:
        fail(f"oracle no longer recovers planted HM*={planted}")
    shf = {r["shipped_mean_error_f1"] for r in rec}
    if len(shf) == 1:
        ok(f"shipped per-class mean flat at {shf.pop():.3f} across the sweep (blind)")
    else:
        fail(f"shipped per-class mean is no longer flat: {sorted(shf)}")
    b1 = orc["experiment_B"]["system1"]; b2 = orc["experiment_B"]["system2"]
    if (abs(b1["ship"]["_mean_error_f1"] - b2["ship"]["_mean_error_f1"]) < 1e-12
            and abs(b1["dec"]["hm"] - b2["dec"]["hm"]) > 0.3):
        ok("oracle witness: identical per-class F1, HM %.4f vs %.4f"
           % (b1["dec"]["hm"], b2["dec"]["hm"]))
    else:
        fail("oracle non-identifiability witness no longer holds")
    assert_in("planted HM", "$\\mathrm{HM}^{*}=%.3f$" % planted)
    assert_in("shipped mean blind to HM",
              "stays $%.3f$" % rec[0]["shipped_mean_error_f1"])

    # Tie-break sensitivity: the matcher's pitch term is bounded by 1e-9 s, and
    # negating it moves HM by <=0.002 with |M| unchanged. Derived from the run.
    tb = _load(os.path.join(HERE, "results", "cluster", "tiebreak_test.json"))
    dmax = 0.0
    for k, v in tb.items():
        if k.startswith("_"):
            continue
        if v["n_matched"][0] != v["n_matched"][1]:
            fail(f"tie-break changed |M| for {k}; the letter says it does not")
        dmax = max(dmax, abs(v["hm"][1] - v["hm"][0]))
    if dmax <= 0.002:
        ok(f"tie-break immaterial: max |dHM| = {dmax:.4f} <= 0.002 as stated")
    else:
        fail(f"tie-break moves HM by {dmax:.4f}; the letter claims <= 0.002")
    with open(os.path.join(os.path.dirname(HERE), "proposal", "spl_supplementary_v5.tex"), "r", encoding="utf-8") as fh:
        _stex = re.sub(r"\s+", " ", fh.read())
    if "moves $\\mathrm{HM}$ by $\\le0.002$" in _stex:
        ok("tie-break bound stated in the supplement")
    else:
        fail("supplement lost the tie-break bound sentence")

    # |U|/|K| bootstrap: points must equal tide_bins' conditional exactly, the
    # three intervals must be pairwise disjoint, and the prose must say so.
    uk = _load(os.path.join(HERE, "results", "cluster", "uk_ci.json"))
    _iv = [uk[st]["ci95"] for st in stems]
    for st in stems:
        c = cv[st]
        K = sum(c[k]["n"] for k in ("extra->wrong", "wrong->wrong", "missed->wrong"))
        U = sum(c[k]["absent_from_score"]
                for k in ("extra->wrong", "wrong->wrong", "missed->wrong"))
        if abs(uk[st]["point"] - U / K) > 1e-12:
            fail(f"uk_ci point for {st} != U/K from adjudication cells")
    disj = all(_iv[i][1] < _iv[i+1][0] or _iv[i+1][1] < _iv[i][0] for i in range(2))
    disj = disj and (_iv[0][1] < _iv[2][0] or _iv[2][1] < _iv[0][0])
    if disj:
        ok("uk_ci: |U|/|K| intervals pairwise disjoint, points match adjudication")
        assert_in("|U|/|K| disjoint clause", "(disjoint per-piece intervals)")
    else:
        fail("uk_ci intervals no longer pairwise disjoint; prose claims they are")

    # Sub-tolerance oracle variant: recovery must include tau=50 ms as printed.
    oj = _load(os.path.join(HERE, "oracle_smalljitter.json"))
    pj = round(oj["planted_hm"], 4)
    if all(r["hm"] == pj for r in oj["sweep"]) and any(
            r["tau_ms"] == 50 for r in oj["sweep"]):
        ok(f"sub-tolerance oracle: HM*={pj} recovered at all "
           f"{len(oj['sweep'])} tolerances incl. 50 ms")
    else:
        fail("sub-tolerance oracle no longer recovers at every tolerance")
    assert_in("oracle recovery clause",
              "recovered exactly at every $\\tau\\ge75$~ms while")

    # Anchor-window sweep of the unfounded share: the ordering claim in the
    # supplement must match the artifact's worst-case flag, and the printed
    # Polytune endpoints must be the derived ones (outward-rounded).
    uw = _load(os.path.join(HERE, "results", "cluster",
                            "unfounded_window_sweep.json"))
    if uw["worst_case_ordering_holds_at_every_window"]:
        ok("unfounded ordering holds at every anchor window (worst-case bounds)")
    else:
        fail("unfounded ordering no longer window-robust; supplement claims it is")
    import math as _m
    lo25, hi25 = uw["per_window"]["w25_exact"]["A_polytune_maestro"]
    lo500, hi500 = uw["per_window"]["w500_exact"]["A_polytune_maestro"]
    _supp2 = os.path.join(os.path.dirname(HERE), "proposal",
                          "spl_supplementary_v5.tex")
    if os.path.exists(_supp2):
        with open(_supp2, encoding="utf-8") as fh:
            s2 = re.sub(r"\s+", " ", fh.read())
        want = ("$%.2f$--$%.2f$ at $25$~ms to $%.2f$--$%.2f$ at $500$~ms"
                % (_m.floor(lo25*100)/100, _m.ceil(hi25*100)/100,
                   _m.floor(lo500*100)/100, _m.ceil(hi500*100)/100))
        if want in s2:
            ok("supplement window-sweep endpoints artifact-derived")
        else:
            fail(f"supplement window-sweep endpoints should read '{want}'")
        # diagonal splits of A and U, derived from the adjudication cells
        Ad = "/".join(str(cv[st]["wrong->wrong"]["in_score_correct"]) for st in stems)
        Ud = "/".join("{:,}".format(cv[st]["wrong->wrong"]["absent_from_score"])
                      .replace(",", "{,}") for st in stems)
        if f"${Ad}$" in s2 and f"${Ud}$" in s2:
            ok("supplement A/U diagonal splits artifact-derived")
        else:
            fail(f"supplement A/U diagonal splits should be ${Ad}$ and ${Ud}$")
        # 0.574 constituents from the replication artifact
        r_u = rep_m["systems"]["laddersym_unprompted"]
        cons = "%.3f/%.3f" % (r_u["pooled_f1_missed"], r_u["pooled_f1_extra"])
        if cons in s2:
            ok(f"supplement Published=0.574 constituents ({cons}) present")
        else:
            fail(f"supplement missing 0.574 constituents {cons}")
        # band instantiation from the printed per-class counts
        T = 7088 + 27650
        X = min(24039, 6888) + min(18829, 15999)
        if ("$T=%s$" % "{:,}".format(T).replace(",", "{,}")) in s2 and            ("%.3f" % (X / (T + X))) in s2:
            ok(f"supplement band instantiated: T={T}, X={X}, X/(T+X)={X/(T+X):.3f}")
        else:
            fail("supplement band instantiation missing or wrong")

    # A-retained charging endpoint (A in the denominator, uncharged) for Polytune
    c0 = cv["A_polytune_maestro"]
    U0 = sum(c0[k]["absent_from_score"] for k in ("extra->wrong", "wrong->wrong", "missed->wrong"))
    M0 = [x for x in _load(os.path.join(gil, "A_polytune_maestro_shipped.json"))
          ["decoupled"]["per_tau"] if x["tau_ms"] == 50][0]
    hm_a = (1173 + 136) / (M0["n_localized"] - U0)
    # post-collapse identification interval: X/(max_k TP_k + X) for Polytune
    s50p = _load(os.path.join(gil, "A_polytune_maestro_shipped.json"))["shipped_50ms"]
    Tmax = max(s50p["missed"]["tp"], s50p["extra"]["tp"])
    Xp = min(s50p["missed"]["fp"], s50p["extra"]["fn"]) + min(s50p["extra"]["fp"], s50p["missed"]["fn"])
    assert_in("post-collapse identification interval", "$[0,%.3f]$ for Polytune" % (Xp / (Tmax + Xp)))
    nc0 = [x for x in _load(os.path.join(gil, "A_polytune_maestro_nocollapse.json"))["decoupled"]["per_tau"] if x["tau_ms"] == 50][0]["hm"]
    Tp = s50p["missed"]["tp"] + s50p["extra"]["tp"]
    ratio0 = (Xp / (Tp + Xp)) / nc0
    assert_in("ambiguity-to-measured ratio", "at least $%d\\times$ that value" % int(ratio0))
    if "It bounds the collapse-free" in tex or "bounds $\\mathrm{HM}_0$" in tex:
        fail("interval described as a bound on HM_0; it is an inner bound on the identified set")
    # Remark 1's old final claim was mathematically false (marginals pin |M| only)
    if "would all be pinned" in tex:
        fail("Remark 1's false 'would all be pinned' claim reintroduced")

    # Bibliographic details verified against Crossref, pinned so a hand-edit
    # cannot silently reintroduce a wrong locator.
    BIB = {}  # DeRA-MOS aside removed 2026-09-01; add DOI->pages pins here
    for doi, page in BIB.items():
        if page in tex:
            ok(f"bib page range for {doi} matches Crossref")
        else:
            fail(f"bib page range for {doi} should be '{page}' (Crossref)")
    if "wang2026deramos" in tex:
        fail("DeRA-MOS aside reintroduced without its Crossref page pin")

    # Verbatim quotations. Each entry is the exact wording of the VERSION WE
    # CITE, verified against the version of record; a paraphrase or a wording
    # taken from a different version of the same paper is a misquote.
    QUOTES = {
        "Polytune (AAAI 2025, p. 23687, Fig. 1 caption)":
            "a missed note and an extra note happening simultaneously",
    }
    for src, q in QUOTES.items():
        if q in tex:
            ok(f"quotation verbatim vs {src}")
        else:
            fail(f"quotation NOT verbatim vs {src}: expected '{q}'")
    if "happening at the same time" in tex:
        fail("Polytune quote uses arXiv v1 wording; the cited AAAI version "
             "reads 'simultaneously'")
    # HOTA's CA extension folds class probability into the matching (Eq. 38);
    # calling its matched set "localization-matched" was refuted at source.
    if "localization-matched set" in tex:
        fail("HOTA mischaracterization reintroduced: its classification-aware "
             "matched set is NOT localization-only (Eq. 38 folds class in)")

    # motivation-vs-evidence gap: the scored corpus under-represents the class
    # the paper is about, relative to the authentic counts it cites as motivation.
    auth, scored = 75 / (75 + 51 + 35), 12258 / 45367
    assert_in("scored substitution share", "$%.1f\\%%$ substitutions" % (scored * 100))
    assert_in("authentic substitution share", "$%.1f\\%%$ ($75/161$)" % (auth * 100))
    assert_in("under-representation factor",
              "a $%.1f\\times$ under-representation" % (auth / scored))

    # 9b-3. anchor-window sweep (the falsifier against a definitional reading)
    rx = _load(os.path.join(HERE, "results", "cluster", "reexam.json"))
    WIN = ["w25_exact", "w500_exact"]
    ctl = [rx["A_polytune_maestro"]["wrong->wrong"][w]["frac_removed"] for w in WIN]
    dom = [rx["A_polytune_maestro"]["extra->wrong"][w]["frac_removed"] for w in WIN]
    if ctl[-1] - ctl[0] >= 0.02:
        fail("anchor sweep: control moved %.4f (artifact drift)" % (ctl[-1] - ctl[0]))
    else:
        ok("anchor sweep: control moves %.4f < 0.02 (artifact)" % (ctl[-1] - ctl[0]))

    # Bootstrap-interval separation, DERIVED not asserted. A blanket "the HM_G
    # intervals overlap" was printed once and was false: the extreme pair is
    # disjoint. This computes the pattern and requires the prose to match it.
    import itertools as _it
    boot = _load(os.path.join(HERE, "results", "cluster", "boot_bins.json"))
    def _disjoint(q, x, y):
        i, j = boot[x]["ci95"][q], boot[y]["ci95"][q]
        return i[0] > j[1] or j[0] > i[1]
    pairs = list(_it.combinations(stems, 2))
    unf_all = all(_disjoint("unfounded", x, y) for x, y in pairs)
    hm_adjacent = [_disjoint("hm_g", stems[0], stems[1]),
                   _disjoint("hm_g", stems[1], stems[2])]
    hm_extreme = _disjoint("hm_g", stems[0], stems[2])
    if unf_all:
        assert_in("unfounded intervals disjoint for every pair",
                  "disjoint for \\emph{every} pair")
    else:
        fail("unfounded intervals are NOT disjoint for every pair; prose says they are")
    if not any(hm_adjacent) and hm_extreme:
        assert_in("HM_G marginal overlap acknowledged",
                  "intervals overlap, yet")
    else:
        fail("HM_G marginal separation pattern changed (adjacent=%s extreme=%s)"
             % (hm_adjacent, hm_extreme))
    # Paired per-piece resampling: every adjacent difference must exclude zero
    # for raw HM, HM_G, and the unfounded share, else the prose overclaims.
    pc_ = _load(os.path.join(HERE, "results", "cluster", "paired_ci.json"))
    for st in stems:
        pt_ = pc_["point"][st]
        if abs(pt_["hm_g"] - boot[st]["point"]["hm_g"]) > 1e-9 or \
           abs(pt_["unfounded"] - boot[st]["point"]["unfounded"]) > 1e-9:
            fail(f"paired_ci point estimates for {st} disagree with boot_bins")
    adj = [f"{stems[0]} - {stems[1]}", f"{stems[1]} - {stems[2]}"]
    all_pos = all(pc_["paired_diff_ci95"][a][k][0] > 0
                  for a in adj for k in ("raw_hm", "hm_g", "unfounded"))
    if all_pos:
        ok("paired resampling: every adjacent difference in raw HM, HM_G, unfounded "
           "excludes zero")
        assert_in("paired ordering clause",
                  "resolves all three orderings (supplementary)")
    else:
        fail("paired adjacent differences no longer all exclude zero; prose says they do")

    # |U|/|M| decomposition: the ordering must not be carried by the merge share
    # alone, so both factors are derived and their monotonicity is checked.
    kM, uK = [], []
    for stem in stems:
        c = cv[stem]
        K = sum(c[k]["n"] for k in ("extra->wrong", "wrong->wrong", "missed->wrong"))
        U = sum(c[k]["absent_from_score"]
                for k in ("extra->wrong", "wrong->wrong", "missed->wrong"))
        t = [x for x in _load(os.path.join(gil, f"{stem}_shipped.json"))
             ["decoupled"]["per_tau"] if x["tau_ms"] == 50][0]
        kM.append(K / t["n_localized"]); uK.append(U / K)
    assert_in("conditional unfounded share",
              "of which $%d/%d/%d\\%%$ are unfounded" % tuple(round(100 * x) for x in uK))
    assert_in("merged share", "$%d/%d/%d\\%%$ of localized events are co-located pairs" % tuple(round(100 * x) for x in kM))
    if uK[0] > uK[1] > uK[2]:
        ok("conditional share |U|/|K| orders the systems independently")
    else:
        fail("|U|/|K| no longer orders the systems; the prose claims it does")

    # the dominant-cell share hull is pinned in section 9b (standard rounding)
    if "by $28\\%$" in tex:
        fail("the unconventioned 'cuts the share by 28%' clause is back")

    # 9c. MAESTRO-EI campaign: every EI number printed in the letter and
    # supplement derives from the campaign artifacts, and the ordering claims
    # (published err-F1 and F rise, HM falls, under every variant and tau)
    # are re-checked from the artifacts, not asserted.
    gei = os.path.join(HERE, "results", "gilbreth_ei")
    ei_cfgs = ("A_polytune_maestro_ei", "B_laddersym_maestro_ei_unprompted",
               "B_laddersym_maestro_ei_prompted")
    summ = _load(os.path.join(HERE, "results", "cluster",
                              "maestro_ei_summary.json"))["summary"]
    n_inj = sum(summ["totals"][k] for k in ("sub", "om", "ins"))
    mix = summ["achieved_mix"]
    inj_str = "{:,}".format(n_inj).replace(",", "{,}")
    assert_in("EI mix and size", "($%.1f\\%%$ substitutions; $%s$ injections)"
              % (mix["substitution"] * 100, inj_str))
    val = _load(os.path.join(HERE, "results", "cluster",
                             "maestro_ei_validate.json"))
    neg_n = val["negative_controls"]["total"]
    neg_m = val["negative_controls"]["merged_by_collapse"]
    if "decoys ($" in tex:
        fail("EI: the vacuous decoy-merge count is printed again; the decoy control "
             "cannot fail by construction (pairs >= 2 s apart, epsilon = 50 ms)")
    # reference-side collapse precision against the manifest (tolerance-matched)
    cp = _load(os.path.join(HERE, "results", "cluster", "maestro_ei_collapse_precision.json"))
    assert_in("EI collapse precision",
              "merges $%s$ pairs, $%.1f\\%%$ of them manifest substitutions (recall $%.4f$ of the"
              % ("{:,}".format(cp["n_reference_merges"]).replace(",", "{,}"),
                 100 * cp["precision"], cp["recall"]))
    assert_in("EI false-merge rate", "the $%.1f\\%%$ remainder is coincidental"
              % (100 * (1 - cp["precision"])))
    if cp["merges_that_are_substitutions"] != round(cp["precision"] * cp["n_reference_merges"]):
        fail("EI collapse precision artifact internally inconsistent")
    if "exactkey" in val["_provenance"]["run"] or val["flippable"] > 200000:
        fail("EI validator JSON is the superseded exact-key run (flippable %d)" % val["flippable"])
    assert_in("EI adjudication ceiling",
              "$%.3f$ of merges at substitution sites"
              % val["adjudication"]["genuine_rate"])
    cvei = {}
    for suf in ("A", "Bu", "Bp"):
        cvei.update(_load(os.path.join(gei, "collapse_validation_ei_%s.json" % suf)))
    ww = [cvei[c]["wrong->wrong"]["manifest_genuine_rate"] for c in ei_cfgs]
    ew = [cvei[c]["extra->wrong"]["manifest_genuine_rate"] for c in ei_cfgs]
    mw = [cvei[c]["missed->wrong"]["manifest_genuine_rate"] for c in ei_cfgs]
    _cp = _load(os.path.join(HERE, "results", "cluster", "maestro_ei_collapse_precision.json"))
    site = [cvei[c]["wrong->wrong"]["manifest_genuine"] / _cp["merges_that_are_substitutions"] for c in ei_cfgs]
    assert_in("EI site-level recall", "i.e.\\ $%.2f/%.2f/%.2f$ of all planted sites" % tuple(site))
    assert_in("collapse absorption in letter", "the collapse absorbs $1{,}112$ of $10{,}317$ planted class flips")
    assert_in("EI genuine contrast",
              "deleted note---on $%.3f/%.3f/%.3f$, i.e." % tuple(ww))
    assert_in("EI dominant-cell genuine rate", "($%.3f/%.3f/%.3f$ in the dominant cell, structurally" % tuple(ew))
    hm50, loc50, err50, mc_all = [], [], [], True
    for c in ei_cfgs:
        d = _load(os.path.join(gei, c + "_shipped.json"))
        t = [x for x in d["decoupled"]["per_tau"] if x["tau_ms"] == 50][0]
        hm50.append(t["hm"]); loc50.append(t["localization"]["f1"])
        err50.append(d["shipped_50ms"]["_mean_error_f1"])
        mc_all &= all(x["mass_conserved"] for x in d["decoupled"]["per_tau"])
    assert_in("EI published err-F1",
              "mean error $F_1$ under the published protocol ($%.3f/%.3f/%.3f$)" % tuple(err50))
    assert_in("EI localization F", "$F=%.3f/%.3f/%.3f$" % tuple(loc50))
    # HM with per-piece intervals from the bootstrap artifacts; the intervals
    # must be pairwise disjoint for HM and for F, else the prose overclaims.
    hm_ci, f_ci = [], []
    for c in ei_cfgs:
        b = _load(os.path.join(gei, c + "_strict_eps05.json"))["bootstrap"]["50"]
        if abs(b["point_hm"] - hm50[len(hm_ci)]) > 1e-9:
            fail("EI bootstrap point HM != shipped HM for %s" % c)
        hm_ci.append(b["hm_ci95"]); f_ci.append(b["loc_f1_ci95"])
    assert_in("EI raw HM with intervals",
              "$\\mathrm{HM}=%.3f/%.3f/%.3f$ with per-piece intervals "
              "$[%.3f,%.3f]/[%.3f,%.3f]/[%.3f,%.3f]$"
              % (*hm50, *hm_ci[0], *hm_ci[1], *hm_ci[2]))
    def _pd(iv):
        return all(iv[i][1] < iv[j][0] or iv[j][1] < iv[i][0]
                   for i in range(3) for j in range(i + 1, 3))
    if _pd(hm_ci) and _pd(f_ci):
        ok("EI: HM and F per-piece intervals pairwise disjoint")
    else:
        fail("EI: HM/F intervals overlap; the prose reports them as distinct levels")
    # Direction: HM is the misclassified fraction, so a FALL is an improvement.
    # All three quantities must order the configs the same way (no re-ranking),
    # and the letter must not claim an inversion.
    if (err50[0] < err50[1] < err50[2] and loc50[0] < loc50[1] < loc50[2]
            and hm50[0] > hm50[1] > hm50[2]):
        ok("EI: err-F1 and F rise while HM (misclassified fraction) falls: "
           "orderings agree, no re-ranking")
    else:
        fail("EI: the three orderings no longer agree; the letter says they do")
    for bad in ("orderings invert", "part ways", "carried by localization",
                "misclassification worsens"):
        if bad in tex:
            fail("EI: inversion misreading reintroduced ('%s')" % bad)
    # abstract / conclusion ranges from the adjudication rates

    # EI adjudication bins (score anchor) from results/cluster/boot_bins_ei.json
    bei = _load(os.path.join(HERE, "results", "cluster", "boot_bins_ei.json"))
    kei = ["A_polytune_maestro_ei", "B_laddersym_maestro_ei_unprompted", "B_laddersym_maestro_ei_prompted"]
    hg = [bei[k]["point"]["hm_g"] for k in kei]; hgc = [bei[k]["ci95"]["hm_g"] for k in kei]
    uf = [100 * bei[k]["point"]["unfounded"] for k in kei]; ufc = [[100 * x for x in bei[k]["ci95"]["unfounded"]] for k in kei]
    assert_in("EI adjudicated HM_G", "gives $\\mathrm{HM}_G=%.3f/%.3f/%.3f$ there (intervals overlapping)" % tuple(hg))
    if not (max(c[0] for c in hgc) <= min(c[1] for c in hgc)):
        fail("EI HM_G intervals do not all overlap; the letter says they do")
    assert_in("EI adjudicated unfounded share", "an unfounded share of $%.1f/%.1f/%.1f\\%%$ (disjoint" % tuple(uf))
    if not (ufc[0][0] > ufc[1][1] and ufc[1][0] > ufc[2][1]):
        fail("EI unfounded intervals are not pairwise disjoint in order")
    assert_in("EI abstract genuine range",
              "name it %.2f to %.2f of the time"
              % (min(ww), max(ww)))
    assert_in("EI conclusion genuine range",
              "$%.2f$--$%.2f$ of" % (min(ww), max(ww)))
    # MAESTRO-E paired decomposition, derived from the paired artifact
    pa = _load(os.path.join(gil, "paired_prompted_vs_unprompted.json"))
    mono = True
    for var in ("shipped", "strict_eps05", "strict_eps0", "pitchaware_eps05"):
        seq = []
        for c in ei_cfgs:
            d = _load(os.path.join(gei, "%s_%s.json" % (c, var)))
            pt = d["decoupled"]["per_tau"] if "decoupled" in d else d["per_tau"]
            seq.append({x["tau_ms"]: x["hm"] for x in pt})
        for tau in seq[0]:
            mono &= seq[0][tau] > seq[1][tau] > seq[2][tau]
    if mono and mc_all:
        ok("EI: HM ordering falls under all four variants at every tau; "
           "mass conserved throughout")
    else:
        fail("EI: HM ordering or mass conservation broke (mono=%s mc=%s); "
             "the letter claims both" % (mono, mc_all))
    guards = [_load(os.path.join(gei, c + "_guard.json")) for c in ei_cfgs]
    a_ok = (guards[0]["n_fail"] == 1 and guards[0]["grand_check"]["passed"]
            and os.path.exists(os.path.join(
                gei, "A_polytune_maestro_ei_guard_OVERRIDE.md")))
    b_ok = guards[1]["all_pass"] and guards[2]["all_pass"]
    if a_ok and b_ok:
        ok("EI guards: 177/177 both LadderSym; Polytune 176/177 + pooled, "
           "override documented")
    else:
        fail("EI guard state changed; letter/supplement describe "
             "177/177 + 176/177-with-pooled")
    if os.path.exists(supp):
        with open(supp, encoding="utf-8") as fh:
            s3 = re.sub(r"\s+", " ", fh.read())
        def s_in(label, printed):
            if printed in s3:
                ok("supplement %s: '%s' artifact-derived" % (label, printed))
            else:
                fail("supplement %s: derived '%s' NOT found" % (label, printed))
        s_in("EI seed", "seed $%d$" % val["_provenance"]["seed"])
        s_in("EI achieved mix", "$%.2f/%.2f/%.2f\\%%$ ($%s$"
             % (mix["substitution"] * 100, mix["insertion"] * 100,
                mix["omission"] * 100, inj_str))
        s_in("EI flippable population", "$q=%.1f$ of the $%s$ non-substitution label events"
             % (val["_provenance"]["q_planted"],
                "{:,}".format(val["flippable"]).replace(",", "{,}")))
        s_in("EI planted flips", "planting $%s$ misclassifications"
             % "{:,}".format(val["planted_flips"]).replace(",", "{,}"))
        n_nonsub = summ["totals"]["ins"] + summ["totals"]["om"] + 2 * summ["totals"]["neg"]
        s_in("EI non-substitution event total", "($%s$ insertion, omission, and decoy"
             % "{:,}".format(n_nonsub).replace(",", "{,}"))
        s_in("EI measured off-diagonal", "measured off-diagonal of $%s$"
             % "{:,}".format(val["off_diagonal"]).replace(",", "{,}"))
        s_in("EI reference merges", "merges $%s$ pairs, $%s$ of them manifest"
             % ("{:,}".format(cp["n_reference_merges"]).replace(",", "{,}"),
                "{:,}".format(cp["merges_that_are_substitutions"]).replace(",", "{,}")))
        f_ = val["recovery"]["flip_fate"]
        s_in("EI flip fate",
             "($%s$ matched off-diagonal, $%d$ diagonal, $%s$ collapse-absorbed, $%s$ unmatched)"
             % ("{:,}".format(f_["matched_offdiag"]).replace(",", "{,}"),
                f_["matched_diag"],
                "{:,}".format(f_["merged"]).replace(",", "{,}"),
                "{:,}".format(f_["unmatched"]).replace(",", "{,}")))
        for lbl, cell in (("EI w->w cell counts", "wrong->wrong"),
                          ("EI e->w cell counts", "extra->wrong")):
            s_in(lbl, "$%s$" % "/".join(
                "{:,}".format(cvei[c][cell]["n"]).replace(",", "{,}")
                for c in ei_cfgs))
        s_in("EI missed->wrong rates",
             "manifest-genuine on $%.3f/%.3f/%.3f$" % tuple(mw))
        s_in("EI guard counts", "passes $%d/%d$" %
             (guards[1]["n_pass"], guards[1]["n_pieces"]))
        s_in("EI Polytune guard", "$%d/%d$ plus pooled" %
             (guards[0]["n_pass"], guards[0]["n_pieces"]))

    # 9d. Round-10 discharges: every added number derives from an artifact.
    # Prop. 1 interval X/(T+X) per configuration from the published-protocol counts
    xt = []
    for stem in stems:
        s50 = _load(os.path.join(gil, f"{stem}_shipped.json"))["shipped_50ms"]
        m_, e_ = s50["missed"], s50["extra"]
        T_ = m_["tp"] + e_["tp"]
        X_ = min(m_["fp"], e_["fn"]) + min(e_["fp"], m_["fn"])
        xt.append(X_ / (T_ + X_))
    assert_in("Prop. 1 interval per configuration",
              "$X/(T{+}X)=%.3f/%.3f/%.3f$" % tuple(xt))
    # pitch-blind bound: raw HM times the equal-pitch share of the off-diagonal
    eqs, bnd = [], []
    for tag in ("polytune", "laddersym_unprompted", "laddersym_prompted"):
        st = _load(os.path.join(HERE, "results", "revision",
                                f"M2_pitch_{tag}.json"))["stratified"]
        eq = st["overall_off_diagonal"]["equal_pitch"]["fraction"]
        eqs.append(eq * 100); bnd.append(st["hm_observed"] * eq)
    assert_in("equal-pitch shares", "$%.1f/%.1f/%.1f\\%%$ of off-diagonal and" % tuple(eqs))
    assert_in("pitch-blind HM bound", "$\\mathrm{HM}\\ge%.3f/%.3f/%.3f$" % tuple(bnd))
    if not (bnd[0] > bnd[1] > bnd[2]):
        fail("pitch-blind bound no longer preserves the ordering; the letter says it does")
    # anchor-window level in the main text, outward-rounded like the supplement
    assert_in("anchor level in letter",
              "$%.2f$--$%.2f$ at $25$~ms to $%.2f$--$%.2f$ at $500$~ms"
              % (_m.floor(lo25*100)/100, _m.ceil(hi25*100)/100,
                 _m.floor(lo500*100)/100, _m.ceil(hi500*100)/100))
    # span endpoints consistent: the raw upper endpoint everywhere
    hm_raw_poly = round([x for x in _load(os.path.join(gil, "A_polytune_maestro_shipped.json"))
                         ["decoupled"]["per_tau"] if x["tau_ms"] == 50][0]["hm"], 3)
    assert_in("conclusion span upper endpoint", "$0.063$ to $%.3f$ for the weakest" % hm_raw_poly)
    if "$0.063$ to $0.242$" in tex:
        fail("conclusion still prints the charge-U endpoint 0.242 as the span's upper end")
    # replication clause: Ladder-row residual from the macro artifact
    mac_u = _load(os.path.join(gil, "replication_macro.json"))["systems"]["laddersym_unprompted"]
    lad_gap = max(abs(mac_u["macro_f1_missed"] - 0.460), abs(mac_u["macro_f1_extra"] - 0.820))
    assert_in("Ladder-row residual", "$%.3f$ for the row printed" % (_m.ceil(lad_gap*1000)/1000))
    # The systems' released evaluator on our predictions must equal the
    # backward-compatible per-piece (macro) values to 4 decimals, class by class.
    rel = _load(os.path.join(gil, "released_evaluator.json"))
    macro_all = _load(os.path.join(gil, "replication_macro.json"))["systems"]
    pairs_ = (("A_polytune_maestro", "polytune"),
              ("B_laddersym_maestro_unprompted", "laddersym_unprompted"),
              ("B_laddersym_maestro_prompted", "laddersym_prompted"))
    worst = 0.0
    for st, sysname in pairs_:
        r_ = rel[st]; m_ = macro_all[sysname]
        worst = max(worst, abs(r_["Track 1 F1"] - m_["macro_f1_missed"]),
                    abs(r_["Track 0 F1"] - m_["macro_f1_extra"]))
        if r_["n_pieces"] != 177:
            fail(f"released evaluator scored {r_['n_pieces']} pieces for {st}, not 177")
    if worst < 5e-5:
        ok(f"released evaluator == backward-compatible per-piece values (max |diff| {worst:.2e})")
        assert_in("released-evaluator clause",
                  "reproduces the backward-compatible mode's per-piece values to four decimals")
    else:
        fail(f"released evaluator differs from backward-compatible values by {worst:.4f}; "
             "the letter claims four-decimal agreement")
    if os.path.exists(supp):
        with open(supp, encoding="utf-8") as fh:
            s4 = re.sub(r"\s+", " ", fh.read())
        want = "%.3f/%.3f" % (mac_u["macro_f1_missed"], mac_u["macro_f1_extra"])
        if want in s4:
            ok(f"supplement prints the unprompted per-piece replication {want}")
        else:
            fail(f"supplement must print the unprompted per-piece replication {want}")
        ok("released-evaluator agreement stated in the letter (pinned there)")
    # the Polytune quotation is pinned to its location in the cited version
    # collapse-free functional: scored without the collapse, must sit inside
    # Prop. 1's interval for every configuration, with the printed CI bound.
    nc, hw = [], 0.0
    for stem, bound in zip(stems, xt):
        d = _load(os.path.join(gil, f"{stem}_nocollapse.json"))
        t = [x for x in d["decoupled"]["per_tau"] if x["tau_ms"] == 50][0]
        b = d["bootstrap"]["50"]
        nc.append(t["hm"]); hw = max(hw, b["hm_ci95"][1] - t["hm"], t["hm"] - b["hm_ci95"][0])
        if not (0.0 <= t["hm"] <= bound):
            fail(f"collapse-free HM {t['hm']:.4f} for {stem} outside Prop. 1 interval [0,{bound:.3f}]")
        if not all(x["mass_conserved"] for x in d["decoupled"]["per_tau"]):
            fail(f"collapse-free arm {stem}: mass conservation failed")
    assert_in("collapse-free HM", "puts at $%.3f/%.3f/%.3f$" % tuple(nc))
    if hw <= 0.007:
        ok(f"collapse-free per-piece intervals within +/-{hw:.4f} <= printed 0.007")
    else:
        fail(f"collapse-free CI half-width {hw:.4f} exceeds printed 0.007")
    if os.path.exists(supp):
        with open(supp, encoding="utf-8") as fh:
            s5 = re.sub(r"\s+", " ", fh.read())
        if ("gives $%.3f/%.3f/%.3f$, inside" % tuple(nc)) in s5:
            ok("supplement collapse-free arm artifact-derived")
        else:
            fail("supplement collapse-free arm values should be %.3f/%.3f/%.3f" % tuple(nc))
    # Polytune's N at 50 ms printed as a smallmatrix; error-density ratio EI/E
    cs0 = M0["confusion_sparse"]
    rows = [[cs0[f"{r}->{c}"] for c in ("missed", "extra", "wrong")] for r in ("missed", "extra", "wrong")]
    off_all = sum(cs0[f"{r}->{c}"] for r in ("missed", "extra", "wrong") for c in ("missed", "extra", "wrong") if r != c)
    mat_b = "\\\\".join("&".join("{:,}".format(cs0[f"{r}->{c}"]).replace(",", "{,}") for c in ("missed", "extra", "wrong")) for r in ("missed", "extra", "wrong"))
    _bb = _load(os.path.join(HERE, "results", "cluster", "boot_bins.json"))
    _hg = [_bb[k]["point"]["hm_g"] for k in ("A_polytune_maestro", "B_laddersym_maestro_unprompted", "B_laddersym_maestro_prompted")]
    assert_in("HM_G spread (letter, twice)", "differs by at most $%.3f$" % (max(_hg) - min(_hg)))
    assert_in("Polytune N bmatrix", "\\begin{bmatrix}" + mat_b + "\\end{bmatrix}")
    _rows = {r: sum(cs0.get(f"{r}->{c}", 0) for c in ("missed", "extra", "wrong")) for r in ("missed", "extra", "wrong")}
    _gam = (23087 - 12258, 34538 - 12258, 12258)  # post-collapse reference totals (gamma_m - mu_r, gamma_e - mu_r, mu_r)
    _rec = [100 * _rows["missed"] / _gam[0], 100 * _rows["extra"] / _gam[1], 100 * _rows["wrong"] / _gam[2]]
    assert_in("N row localization rates", "localize $%.1f\\%%$ of pure omissions against $%.1f\\%%$ of insertions and $%.1f\\%%$ of substitutions" % tuple(_rec))
    assert_in("Polytune dominant cell", "holds ${:,}$ of ${:,}$ off-diagonal events".format(cs0["extra->wrong"], off_all).replace(",", "{,}"))
    # pre-collapse reference label events, EI over E (the basis the letter names)
    dens = val["n_ref_events"] / (45367 + 12258)
    assert_in("EI/E reference-label ratio",
              "and $%.1f\\times$ the reference label events." % dens)
    ap_ = _load(os.path.join(HERE, "results", "cluster", "anchor_pitch_sensitivity.json"))
    semi = [100 * ap_[st]["unfounded_share"]["semitone"] for st in stems]
    octv = [100 * ap_[st]["unfounded_share"]["octave"] for st in stems]
    exact = [100 * ap_[st]["unfounded_share"]["exact"] for st in stems]
    if not all(abs(exact[i] - 100 * unf[i]) < 0.05 for i in range(3)):
        fail("anchor_pitch_sensitivity exact-pitch shares disagree with tide_bins")
    with open(os.path.join(os.path.dirname(HERE), "proposal", "spl_supplementary_v5.tex"), "r", encoding="utf-8") as fh:
        _stex2 = re.sub(r"\s+", " ", fh.read())
    _pt = "($%.1f/%.1f/%.1f\\%%$ and $%.1f/%.1f/%.1f\\%%$)" % (*semi, *octv)
    if _pt in _stex2:
        ok(f"supplement pitch-tolerant unfounded shares: '{_pt}'")
    else:
        fail(f"supplement pitch-tolerant shares: derived '{_pt}' NOT found")
    if not (semi[0] > semi[1] > semi[2] and octv[0] > octv[1] > octv[2]):
        fail("pitch-tolerant unfounded shares no longer preserve the ordering")
    # share of U that a pitch-tolerant anchor re-founds, over both tolerances
    frac_moved = [100 * (1 - v[i] / exact[i]) for v in (semi, octv) for i in range(3)]
    assert_in("near-miss share of U",
              "falls $%d$--$%d\\%%$ under"
              % (round(min(frac_moved)), round(max(frac_moved))))
    diag = [100 * ap_[st]["diag_equal_pitch_frac"] for st in stems]
    assert_in("diagonal equal-pitch share", "$%.1f/%.1f/%.1f\\%%$ of diagonal pairs join equal" % tuple(diag))
    if "larger by construction" in tex:
        fail("'larger by construction' reintroduced; the collapse's sign is not a theorem")
    assert_in("quote location", "simultaneously'' (Fig.~1 of~")
    assert_in("LadderSym availability wording", "not released; accessed 1~Sep.")


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
    check_score_filter()
    check_round17_supplement()
    check_figs_tables()
    check_letter_prose()
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
