# Experimental research source — 2026-10-29

Diffusion research prototype. Reachable history reviewed for source publication; no dataset,
checkpoint, credentials or private results were committed.

Restored only the missing Python CIFAR loader modules under datasets/ from an older local staging
copy. Their syntax and caller signatures were reviewed; no dataset or weights were downloaded.

KNOWN BROKEN — train.py retains its existing indentation/syntax errors. Three scoped attempts
could not establish a safe indentation-only repair, so that file is unchanged. Do not treat this
as a runnable training release or a verified research result. Other loaders/modules parse, but
full training, GPU behavior and quality/speed claims were NOT VERIFIED.
