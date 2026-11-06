#!/usr/bin/env python3
"""Vendor the Windows ONNX Runtime DLL and the pinned VoxLock voice models.

The release workflow runs this before packaging, so the downloaded ZIP needs no
setup script, no Python, and no first-launch model download. Files are verified
against the hashes below and reused when they already match.

    python scripts/vendor.py            # runtime + models
    python scripts/vendor.py check      # verify only
    python scripts/vendor.py package    # build the release ZIP
    python scripts/vendor.py runtime --force   # replace a CUDA runtime

The ONNX Runtime build is the official 1.30.0 NuGet package; the models come
from the VoxLock revision pinned in models.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RUNTIME = ROOT / "runtime"
CACHE = ROOT / "target" / "vendor"
MANIFEST_PATH = ROOT / "models.json"

ORT_VERSION = "1.30.0"
ORT_NUPKG_URL = (
    f"https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/{ORT_VERSION}"
)
ORT_NUPKG_SHA256 = "30df119bd9164eaa187666d93147448fdd4a9a1a3caf83f1505daf7af2d1c5d2"
ORT_DLL_MEMBER = "runtimes/win-x64/native/onnxruntime.dll"
ORT_DLL_SHA256 = "7e39e2bdbba836d98071ef28620735ba36a47c554cf794585269aecc50fab0da"
ORT_DLL = RUNTIME / "onnxruntime.dll"

CHUNK = 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def matches(path: Path, expected_sha256: str, expected_bytes: int | None = None) -> bool:
    if not path.is_file():
        return False
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        return False
    return sha256_file(path) == expected_sha256


def load_manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def download(url: str, destination: Path, expected_sha256: str, expected_bytes: int | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "beeswax-vendor/1.0"})
    digest = hashlib.sha256()
    received = 0
    last_report = 0.0
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            header = response.headers.get("Content-Length")
            total = int(header) if header and header.isdigit() else None
            while True:
                block = response.read(256 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
                received += len(block)
                now = time.monotonic()
                if now - last_report >= 2.0 or (total is not None and received == total):
                    if total:
                        print(
                            f"  {received / 1_000_000:.1f} / {total / 1_000_000:.1f} MB",
                            end="\r",
                            flush=True,
                        )
                    else:
                        print(f"  {received / 1_000_000:.1f} MB", end="\r", flush=True)
                    last_report = now
    except (urllib.error.URLError, OSError) as error:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"Download failed: {error}") from error
    print()
    if expected_bytes is not None and received != expected_bytes:
        partial.unlink(missing_ok=True)
        raise SystemExit(
            f"Downloaded {destination.name} has {received} bytes, expected {expected_bytes}"
        )
    actual = digest.hexdigest()
    if actual != expected_sha256:
        partial.unlink(missing_ok=True)
        raise SystemExit(
            f"SHA-256 mismatch for {destination.name}: got {actual}, expected {expected_sha256}"
        )
    os.replace(partial, destination)


def ensure_runtime(force: bool = False) -> None:
    if not force and matches(ORT_DLL, ORT_DLL_SHA256):
        print("runtime: onnxruntime.dll is up to date")
        return
    if ORT_DLL.exists() and not force:
        raise SystemExit(
            "runtime/onnxruntime.dll exists but is not the pinned CPU build "
            f"(ONNX Runtime {ORT_VERSION}). It may be a CUDA runtime; pass --force "
            "to replace it with the CPU build."
        )

    CACHE.mkdir(parents=True, exist_ok=True)
    nupkg = CACHE / f"Microsoft.ML.OnnxRuntime.{ORT_VERSION}.nupkg"
    if not matches(nupkg, ORT_NUPKG_SHA256):
        nupkg.unlink(missing_ok=True)
        download(ORT_NUPKG_URL, nupkg, ORT_NUPKG_SHA256, None)
    else:
        print(f"runtime: reusing {nupkg.relative_to(ROOT)}")

    print(f"Extracting {ORT_DLL_MEMBER} from ONNX Runtime {ORT_VERSION}")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    temporary = ORT_DLL.with_name(ORT_DLL.name + ".tmp")
    digest = hashlib.sha256()
    try:
        with zipfile.ZipFile(nupkg) as archive, archive.open(ORT_DLL_MEMBER) as source:
            with temporary.open("wb") as output:
                for block in iter(lambda: source.read(CHUNK), b""):
                    output.write(block)
                    digest.update(block)
    except (KeyError, zipfile.BadZipFile) as error:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"Could not extract {ORT_DLL_MEMBER} from {nupkg.name}: {error}") from error
    actual = digest.hexdigest()
    if actual != ORT_DLL_SHA256:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"Extracted onnxruntime.dll has SHA-256 {actual}, expected {ORT_DLL_SHA256}"
        )
    os.replace(temporary, ORT_DLL)
    print(f"runtime: installed {ORT_DLL.relative_to(ROOT)}")


def ensure_models(force: bool = False) -> None:
    manifest = load_manifest()
    repository = manifest["repository"]
    revision = manifest["revision"]
    MODELS.mkdir(parents=True, exist_ok=True)
    for model in manifest["models"]:
        name = model["name"]
        destination = MODELS / name
        expected_bytes = int(model["bytes"])
        if not force and matches(destination, model["sha256"], expected_bytes):
            print(f"models: {name} is up to date")
            continue
        if destination.exists() and not force:
            raise SystemExit(
                f"models/{name} does not match models.json. Pass --force to download "
                "a clean copy."
            )
        url = f"https://huggingface.co/{repository}/resolve/{revision}/{model['path']}"
        download(url, destination, model["sha256"], expected_bytes)
        print(f"models: installed {name} ({expected_bytes / 1_000_000:.1f} MB)")


def check() -> bool:
    problems = []
    if not matches(ORT_DLL, ORT_DLL_SHA256):
        problems.append(
            "runtime/onnxruntime.dll is missing or does not match the pinned "
            f"ONNX Runtime {ORT_VERSION} build"
        )
    for model in load_manifest()["models"]:
        path = MODELS / model["name"]
        if not matches(path, model["sha256"], int(model["bytes"])):
            problems.append(f"models/{model['name']} is missing or has the wrong checksum")
    if problems:
        for problem in problems:
            print(f"vendor: {problem}")
        print("Run 'python scripts/vendor.py' to install the missing files.")
        return False
    print("vendor: ONNX Runtime and voice models match their pinned checksums")
    return True


def package(binary: Path, output: Path) -> None:
    if not check():
        raise SystemExit("Refusing to package an incomplete build.")
    if not binary.is_file():
        raise SystemExit(f"Binary not found: {binary}. Run 'cargo build --release --locked' first.")

    entries: list[tuple[Path, str, int]] = [
        (binary, "beeswax.exe", zipfile.ZIP_DEFLATED),
        (ROOT / "LICENSE", "LICENSE", zipfile.ZIP_DEFLATED),
        (ROOT / "README.md", "README.md", zipfile.ZIP_DEFLATED),
        (MANIFEST_PATH, "models.json", zipfile.ZIP_DEFLATED),
        (ORT_DLL, "runtime/onnxruntime.dll", zipfile.ZIP_STORED),
    ]
    for model in load_manifest()["models"]:
        entries.append((MODELS / model["name"], f"models/{model['name']}", zipfile.ZIP_STORED))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source, name, compression in entries:
            archive.write(source, name, compress_type=compression)
    os.replace(temporary, output)
    print(f"package: {output.relative_to(ROOT)} ({output.stat().st_size / 1_000_000:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("all", "runtime", "models", "check", "package"),
        help="what to vendor (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="replace existing files")
    parser.add_argument(
        "--binary",
        type=Path,
        default=ROOT / "target" / "release" / "beeswax.exe",
        help="release binary to package",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "target" / "beeswax-windows-x64.zip",
        help="ZIP path for the package command",
    )
    args = parser.parse_args()

    if args.command in ("all", "runtime"):
        ensure_runtime(args.force)
    if args.command in ("all", "models"):
        ensure_models(args.force)
    if args.command == "check" and not check():
        sys.exit(1)
    if args.command == "package":
        package(args.binary, args.output)


if __name__ == "__main__":
    main()
