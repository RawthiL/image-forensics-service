# stub

**Not a model.** Hashes pixel bytes into a score in [0, 1].

| | |
|---|---|
| Weights | none |
| Dependencies | none beyond Pillow |
| License | this repo's |

Exists so CI, the Docker healthcheck, and anyone wiring up a dialect can
exercise every route without torch, a GPU, or a download. Scores are
deterministic across runs and completely meaningless.

**Never enable in production.** `IFS_DETECTOR=stub` is a test fixture.
