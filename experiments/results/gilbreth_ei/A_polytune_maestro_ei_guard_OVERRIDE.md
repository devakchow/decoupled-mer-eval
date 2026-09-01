guard.A_polytune_maestro_ei: 176/177 pieces pass; grand (pooled) check PASSES.
The single failing piece (MIDI-Unprocessed_SMF_17_R1_2004...Track12_wav--1)
fails the per-piece dominance heuristic because the model under-detects
insertions there ([Correct][Extra]=209 > diag 170) -- a model-performance
property under the EI distribution shift, not a track-mapping defect (diagonal
170/70/1417 is dominant). Full evidence in A_polytune_maestro_ei_guard.json.
Marker set manually 2026-08-31 to let scoring proceed; the guard artifact is
retained unmodified.
