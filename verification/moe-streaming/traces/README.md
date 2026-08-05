# Runtime trace location

`Trace.tla` reads `trace.ndjson` here by default. Override it per TLC run with
`-DJSON=/absolute/path/to/trace.ndjson` (exposed through `IOEnv.JSON`).

Trace harness instrumentation is intentionally specified, but not compiled into
normal llama.cpp builds; see `../instrumentation-spec.md`. Focused GPU tests are
the current implementation-to-model validation evidence. A later harness pass
should emit short (50–300 event), per-thread timebox traces from those scenarios.
