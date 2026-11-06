#!/bin/sh
set -eu
cd "$(dirname "$0")"

# The competing-talker fixture is deterministic and cached; regenerate only if
# it is missing (e.g. after cleaning target/).
if [ ! -f target/autoresearch/interference.wav ]; then
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts/bench-fixture.ps1 >&2
fi

cargo build --release --locked --bin beeswax-bench --quiet
exec ./target/release/beeswax-bench.exe
