#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_gt_meta.py -- build the ground-truth metadata JSON the decoupled scorer,
dominance guard, and shipped-mode regression all consume.

Same construction as the smoke's ``smoke_gt_meta.json`` (wsl_smoke/t5_make_meta.sh)
but for ALL test ids, both datasets:

  MAESTRO-E (``--dataset maestro``)
    key   = <id>  (directory name under label/<type>/)
    value = {extra_notes_midi, removed_notes_midi, correct_notes_midi}
    exactly ONE label/<type>/<id>/stems_midi/*.mid per type (asserted).

  CocoChorales-E (``--dataset coco``)
    key   = <id>_<stem>   e.g. "216001_0_flute"
    stem  = basename (sans .mid) of label/<type>/<id>/stems_midi/<n>_<inst>.mid
            (lowercase-underscored on disk; asserted identical across the
            three label types per id).
    value = the three per-stem label paths.

This keying matches flatten_coco_preds.py, which names each bridged coco
prediction <id>_<normalized-stem>.mid where the normalization is exactly the
shipped eval's own rule (lower(), ' ' -> '_'); e.g. prediction
"216010/1_French Horn.mid" -> key "216010_1_french_horn".

Fail-loud: any missing/extra/ambiguous label file aborts with a message that
names the id. ``--expect N`` additionally asserts the final key count.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

LABEL_TYPES = [("extra_notes_midi", "extra_notes"),
               ("removed_notes_midi", "removed_notes"),
               ("correct_notes_midi", "correct_notes")]


def ids_of(root: str, sub: str) -> list:
    d = os.path.join(root, "label", sub)
    if not os.path.isdir(d):
        sys.exit("gen_gt_meta: missing directory %s" % d)
    return sorted(e for e in os.listdir(d)
                  if os.path.isdir(os.path.join(d, e)))


def build_maestro(root: str) -> dict:
    id_sets = {sub: ids_of(root, sub) for _, sub in LABEL_TYPES}
    ids = id_sets["extra_notes"]
    for sub, got in id_sets.items():
        if got != ids:
            sys.exit("gen_gt_meta: id set mismatch label/%s vs label/extra_notes "
                     "(%d vs %d ids)" % (sub, len(got), len(ids)))
    meta = {}
    for i in ids:
        entry = {}
        for key, sub in LABEL_TYPES:
            hits = sorted(glob.glob(
                os.path.join(root, "label", sub, i, "stems_midi", "*.mid")))
            if len(hits) != 1:
                sys.exit("gen_gt_meta: %s/%s: expected exactly 1 label .mid, "
                         "found %d: %s" % (sub, i, len(hits), hits))
            entry[key] = hits[0]
        meta[i] = entry
    return meta


def build_coco(root: str) -> dict:
    id_sets = {sub: ids_of(root, sub) for _, sub in LABEL_TYPES}
    ids = id_sets["extra_notes"]
    for sub, got in id_sets.items():
        if got != ids:
            sys.exit("gen_gt_meta: id set mismatch label/%s vs label/extra_notes "
                     "(%d vs %d ids)" % (sub, len(got), len(ids)))
    meta = {}
    for i in ids:
        per_type = {}
        for key, sub in LABEL_TYPES:
            hits = sorted(glob.glob(
                os.path.join(root, "label", sub, i, "stems_midi", "*.mid")))
            if not hits:
                sys.exit("gen_gt_meta: no label mids for %s/%s" % (sub, i))
            per_type[key] = {os.path.splitext(os.path.basename(h))[0]: h
                             for h in hits}
        stems = sorted(per_type["extra_notes_midi"])
        for key, _ in LABEL_TYPES:
            got = sorted(per_type[key])
            if got != stems:
                sys.exit("gen_gt_meta: %s: stem set mismatch %s vs %s for id %s"
                         % (key, got, stems, i))
        for stem in stems:
            k = "%s_%s" % (i, stem)
            if k in meta:
                sys.exit("gen_gt_meta: duplicate key %s" % k)
            meta[k] = {key: per_type[key][stem] for key, _ in LABEL_TYPES}
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, choices=("maestro", "coco"))
    ap.add_argument("--root", required=True, help="dataset root (contains label/)")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--expect", type=int, default=None,
                    help="assert exactly this many keys")
    a = ap.parse_args(argv)

    meta = build_maestro(a.root) if a.dataset == "maestro" else build_coco(a.root)
    if a.expect is not None and len(meta) != a.expect:
        sys.exit("gen_gt_meta: built %d keys, expected %d -- dataset "
                 "incomplete or layout drifted; refusing to write %s"
                 % (len(meta), a.expect, a.out))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    print("gen_gt_meta: wrote %s  dataset=%s  keys=%d"
          % (a.out, a.dataset, len(meta)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
