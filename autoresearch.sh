#!/bin/sh
set -eu
cd "$(dirname "$0")"

# The autoresearch runner invokes this through WSL on some hosts. Windows tools
# are reachable there through interop; elsewhere they are on PATH directly.
if [ -r /proc/version ] && grep -qi microsoft /proc/version 2>/dev/null; then
    CARGO="/mnt/c/Windows/System32/cmd.exe /c cargo"
    POWERSHELL="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
else
    CARGO=cargo
    POWERSHELL=powershell
fi

# The competing-talker fixture is deterministic and cached; regenerate only if
# it is missing (e.g. after cleaning target/).
if [ ! -f target/autoresearch/interference.wav ]; then
    $POWERSHELL -NoProfile -ExecutionPolicy Bypass -File scripts/bench-fixture.ps1 >&2
fi

$CARGO build --release --locked --bin beeswax-bench --quiet
exec ./target/release/beeswax-bench.exe
