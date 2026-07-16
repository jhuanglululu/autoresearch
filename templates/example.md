# Goal: (example — replace me)

Optimize inference speed of a ~50M-param decoder-only LM trained on the pinned
wikipedia-zhtw corpus, without degrading val loss by more than 2% relative to the
baseline architecture.

## Constraints
- Architecture-level ideas only; training-dynamics tweaks are the engineer's discretion.
- Every claim in the wiki must cite a source (paper, page, or experiment record).
- One experiment at a time on the single GPU.

## Success criteria
- A summary note per tested idea with a clear verdict (adopted / rejected / inconclusive),
  citing its experiment run records.
