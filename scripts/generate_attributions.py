# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate ATTRIBUTIONS.md by invoking pip-licenses directly.

Install with `pip install pip-licenses`.

Run this inside an environment where the target packages are installed
(typically the release container) so pip-licenses can read each package's
`.dist-info/` metadata.

Workflow:
  1. Parse package names from `--requirements` (a pip requirements .txt).
  2. Call `python -m piplicenses --packages <names> --with-urls
     --with-authors --format=json` as a subprocess.
  3. For each github.com project URL, resolve the canonical LICENSE-file
     URL via the GitHub REST API (`/repos/{owner}/{repo}/license`).
     Non-GitHub URLs are kept as-is. Pass `--no-resolve` to skip.
  4. Emit one section per package, alphabetised, in the format:

         ## Copyright <Author> - `<Name>` - <License>
         License Text(<URL>)
         <LicenseText>

Only four pip-licenses fields are used: Name, Author, License, URL.

Example (from the container):
    python3 scripts/generate_attributions.py \\
        --requirements requirements.txt \\
        --output ATTRIBUTIONS.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

GIT_URL_TO_PACKAGE: dict[str, str] = {
    "sam2": "SAM-2",
}


HEADER = """\
# Open Source License Attribution

This project uses Open Source components. You can find the details of these open-source projects along with license information below, sorted alphabetically.
We are grateful to the developers for their contributions to open source and acknowledge these below.
Source code for all open-source binaries distributed in the container are available in `/usr/share/oss-sources/` and `/usr/lib/python3/dist-packages/`.

"""


def extract_name(spec: str) -> str | None:
    spec = spec.strip()
    if not spec or spec.startswith(("#", "-")):
        return None
    if "git+" in spec:
        m = re.search(r"/([A-Za-z0-9_.-]+?)(?:\.git)?(?:@|$|#)", spec)
        if not m:
            return None
        repo = m.group(1)
        return GIT_URL_TO_PACKAGE.get(repo.lower(), repo)
    m = re.match(r"^([A-Za-z0-9_.-]+)", spec)
    return m.group(1) if m else None


def parse_requirements(path: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for line in path.read_text().splitlines():
        n = extract_name(line)
        if n and n.lower() not in seen:
            seen.add(n.lower())
            names.append(n)
    return names


def run_pip_licenses(packages: list[str], with_license_text: bool) -> list[dict]:
    cmd = [
        sys.executable,
        "-m",
        "piplicenses",
        "--packages",
        *packages,
        "--with-urls",
        "--with-authors",
        "--format=json",
    ]
    if with_license_text:
        cmd.append("--with-license-file")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if "No module named" in result.stderr:
            print(
                "error: `piplicenses` is not importable. Install with `pip install pip-licenses` "
                "in the environment where the target packages live (e.g. the release container).",
                file=sys.stderr,
            )
        else:
            print(
                f"error: pip-licenses exited with code {result.returncode}",
                file=sys.stderr,
            )
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode or 2)
    return json.loads(result.stdout)


def clean(value: str | None, fallback: str = "Unknown") -> str:
    if not value or value == "UNKNOWN":
        return fallback
    return value


GITHUB_URL_RE = re.compile(
    r"^https?://github\.com/([^/\s]+)/([^/\s.]+?)(?:/|\.git|$)", re.IGNORECASE
)


def resolve_license_url(
    url: str,
    cache: dict[str, str],
    token: str | None,
    failures: list[str],
) -> str:
    """For a github.com project URL, return the canonical LICENSE-file URL.

    Uses the GitHub REST API (`/repos/{owner}/{repo}/license`) so the result
    points at whatever the project actually uses — handles `main` vs `master`
    and unusual filenames (`COPYING`, `LICENSE.md`, ...). Non-GitHub URLs are
    returned unchanged. API failures are recorded in `failures` and the
    original URL is returned so the run can continue.

    Pass a GitHub token (PAT, `Bearer`) to lift the 60/hr anonymous rate limit
    to 5000/hr — required for any non-trivial run.
    """
    if url in cache:
        return cache[url]
    m = GITHUB_URL_RE.match(url.strip())
    if not m:
        cache[url] = url
        return url
    owner, repo = m.group(1), m.group(2)
    api = f"https://api.github.com/repos/{owner}/{repo}/license"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "cosmos-anomalygen-attributions",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(api, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.load(resp)
        resolved = payload.get("html_url") or url
        # Only cache successful API responses — failures should retry next run.
        cache[url] = resolved
        return resolved
    except urllib.error.HTTPError as e:
        remaining = e.headers.get("X-RateLimit-Remaining") if e.headers else None
        if e.code == 403 and remaining == "0":
            failures.append(f"{owner}/{repo}: rate limit exceeded")
        else:
            failures.append(f"{owner}/{repo}: HTTP {e.code} {e.reason}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        failures.append(f"{owner}/{repo}: {type(e).__name__}")
    return url


def load_cache(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_cache(path: Path | None, cache: dict[str, str]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    except OSError as e:
        print(f"warning: could not write cache to {path}: {e}", file=sys.stderr)


def render(
    data: list[dict],
    resolve: bool,
    cache: dict[str, str],
    token: str | None,
    failures: list[str],
    with_license_text: bool,
) -> str:
    data.sort(key=lambda e: e["Name"].lower())
    parts: list[str] = [HEADER]
    for e in data:
        name = e["Name"]
        author = clean(e.get("Author"), fallback=f"The {name} team")
        license_ = clean(e.get("License"))
        url = clean(e.get("URL"), fallback="")
        if url and resolve:
            url = resolve_license_url(url, cache, token, failures)
        parts.append(f"## Copyright {author} - `{name}` - {license_}\n\n")
        if url:
            parts.append(f"License Text({url})\n")
        if with_license_text:
            text = (e.get("LicenseText") or "").strip()
            if text:
                parts.append("\n")
                parts.append(text + "\n")
        parts.append("\n")
    return "".join(parts).rstrip("\n") + "\n"


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--requirements",
        required=True,
        type=Path,
        help="pip requirements .txt to scope packages",
    )
    ap.add_argument(
        "--output", required=True, type=Path, help="output ATTRIBUTIONS.md path"
    )
    ap.add_argument(
        "--no-resolve",
        action="store_true",
        help="skip the GitHub API lookup for canonical LICENSE-file URLs",
    )
    ap.add_argument(
        "--github-token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub PAT for the license API (lifts rate limit to 5000/hr). "
             "Defaults to $GITHUB_TOKEN.",
    )
    ap.add_argument(
        "--cache",
        type=Path,
        default=Path(".github-license-cache.json"),
        help="JSON cache file for resolved URLs (default: .github-license-cache.json)",
    )
    ap.add_argument(
        "--with-license-text",
        action="store_true",
        help="inline each package's verbatim LICENSE text after the URL line "
             "(matches the OSRB '3rd Party Notice Files Example' format).",
    )
    args = ap.parse_args()

    packages = parse_requirements(args.requirements)
    if not packages:
        print(f"error: no packages parsed from {args.requirements}", file=sys.stderr)
        return 2

    print(
        f"running pip-licenses on {len(packages)} package(s) from {args.requirements}"
    )
    data = run_pip_licenses(packages, with_license_text=args.with_license_text)
    cache = load_cache(args.cache) if not args.no_resolve else {}
    failures: list[str] = []
    if not args.no_resolve:
        auth = "authenticated" if args.github_token else "anonymous (60/hr limit)"
        cached = sum(1 for e in data if (e.get("URL") or "") in cache)
        print(f"resolving LICENSE URLs via GitHub API ({auth}, {cached}/{len(data)} cached) …")
    body = render(
        data,
        resolve=not args.no_resolve,
        cache=cache,
        token=args.github_token,
        failures=failures,
        with_license_text=args.with_license_text,
    )
    if not args.no_resolve:
        save_cache(args.cache, cache)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body)

    not_found = sorted(
        {normalise(p) for p in packages} - {normalise(e["Name"]) for e in data}
    )
    print(f"wrote {args.output} with {len(data)} entries")
    if not_found:
        print(
            f"\nIn requirements but not installed (skipped by pip-licenses) — {len(not_found)}:"
        )
        for n in not_found:
            print(f"  - {n}")
    if failures:
        print(
            f"\nGitHub API resolution failed for {len(failures)} repo(s) "
            "(fell back to project URL):"
        )
        for f in failures:
            print(f"  - {f}")
        if any("rate limit" in f for f in failures):
            print(
                "\nHint: set GITHUB_TOKEN (or pass --github-token) to lift the "
                "anonymous 60/hr limit to 5000/hr."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
