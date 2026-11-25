# Security

MM Engine is a research and education project. It is not connected to any exchange or broker and does not handle money or credentials.

What the engine guarantees:

- Every order is checked before it touches the book. Non-positive sizes, prices outside the configured band and duplicate order ids are rejected (the call returns -1).
- Memory is preallocated. The order pool and id table are sized at construction, so the hot path never allocates, and the id table can never fill up: it holds twice the pool size.
- CI runs the full test suite under AddressSanitizer and UndefinedBehaviorSanitizer on every push, with any finding failing the build.

Known limits:

- The price band is allocated as two arrays of `max_price - min_price + 1` levels. Choose a band that fits in memory.
- `replace()` cancels and then re-adds. If the new order is rejected, the old one is already gone, which matches how most exchanges treat a cancel-replace.

To report a bug or a security issue, open a GitHub issue with steps to reproduce.
