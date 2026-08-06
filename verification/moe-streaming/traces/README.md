# Runtime trace location

`Trace.tla` reads `trace.ndjson` here by default. That file is a recorded
artifact and is not committed, so supply a path explicitly until a harness
produces one. Override it per TLC run with the `JSON` environment variable,
read through `IOEnv.JSON`; a `-DJSON=...` JVM property is not visible to
`IOEnv`. The `Json` and `IOUtils` modules come from the TLA+ CommunityModules.
From `../spec`:

```sh
JSON=/absolute/path/to/trace.ndjson \
java -cp tla2tools.jar:CommunityModules-deps.jar tlc2.TLC \
    -config Trace.cfg Trace.tla
```

`example-decode.ndjson` is a hand-written three-event smoke trace (miss, CPU
completion, node end) that exercises the validator itself. It is not recorded
from the implementation.

A rejected trace shows up in one of two ways. An event whose post-state
contradicts the model enables no action, so TLC reports `Deadlock reached` and
prints the state before the offending event. A log that stalls for any other
reason leaves events unconsumed and violates `TraceFullyConsumed`. Both are
rejections; the deadlock report is the more useful one because it names the
event that failed to match.

Trace harness instrumentation is intentionally specified, but not compiled into
normal llama.cpp builds; see `../instrumentation-spec.md`. Focused GPU tests are
the current implementation-to-model validation evidence. A later harness pass
should emit short (50-300 event), per-thread timebox traces from those scenarios.
