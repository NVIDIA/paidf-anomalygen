#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# oss_bake.sh — collect the third-party source that must travel with the published image.
#
#   archives   pinned tarballs no index serves (apex, flash-attn-3-nv, sam-2) plus uv, CPython and
#              the GCC whose libquadmath the numpy/scipy wheels bundle
#   apt        Debian source for the packages the Dockerfile adds on top of the base image
#   pip        sdist per venv distribution, or the wheel where none is published
#   manifest   sha256 roll-up over the whole bake, for audit — run it last
#
# Runs inside the image being built; the Dockerfile COPYs it to /usr/local/bin/oss-bake. Versions and
# digests come from the Dockerfile PINS block, which declares them as ARGs in the runtime stage, so
# they arrive here as environment variables and keep a single home. `archives` and `pip` also need
# TORCH_INDEX_URL. `apt` consumes and deletes the pre-install dpkg snapshot, so it runs once per
# image; `pip` skips what is already in pip/, which makes the develop second pass a delta.
#
# `set -eu`, NOT `-o pipefail`: several counts are `ls ... | wc -l` over globs that legitimately match
# nothing, and pipefail would turn those into a failed build.
set -eu

usage() {
  awk '/^#!/ {next} /SPDX-/ {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}" >&2
  exit 64
}

# Digest-checked, so MANIFEST certifies verified bytes rather than whatever the network returned.
# ${VAR:?} on every pin: an ARG that is declared but empty would otherwise build a garbage URL.
fetch() {  # fetch <url> <out> <sha256>
  local url="$1" out="$2" sha="$3"
  curl -fsSL --proto '=https' --proto-redir '=https' "$url" -o "$out"
  echo "${sha}  ${out}" | sha256sum -c - >/dev/null
}

bake_archives() {
  D=/usr/share/oss-sources
  mkdir -p "$D/source-builds"
  fetch "https://github.com/NVIDIA/apex/archive/${APEX_SHA:?}.tar.gz" \
        "$D/source-builds/apex-${APEX_SHA}.tar.gz" "${APEX_TARBALL_SHA256:?}"
  fetch "https://github.com/alihassanijr/flash_attn_3_nv/archive/${FA3_SHA:?}.tar.gz" \
        "$D/source-builds/flash_attn_3_nv-${FA3_SHA}.tar.gz" "${FA3_TARBALL_SHA256:?}"
  fetch "https://github.com/astral-sh/uv/archive/refs/tags/${UV_VERSION:?}.tar.gz" \
        "$D/uv-${UV_VERSION}.tar.gz" "${UV_TARBALL_SHA256:?}"
  fetch "https://www.python.org/ftp/python/${PYTHON_VERSION:?}/Python-${PYTHON_VERSION}.tar.xz" \
        "$D/Python-${PYTHON_VERSION}.tar.xz" "${PYTHON_TARBALL_SHA256:?}"
  fetch "https://github.com/facebookresearch/sam2/archive/${SAM2_SHA:?}.tar.gz" \
        "$D/source-builds/sam-2-${SAM2_SHA}.tar.gz" "${SAM2_TARBALL_SHA256:?}"

  # GCC source for the libquadmath the numpy and scipy wheels bundle: LGPL-2.1, outside the
  # libgfortran exception, and absent from their sdists. Mirrors in turn — not every runner
  # reaches ftp.gnu.org. All three failing leaves no file, and the digest check below says so.
  V="${LIBGFORTRAN_LIBQUADMATH_GCC_VER:?}"
  F="$D/gcc-$V.tar.xz"
  for u in "https://ftp.gnu.org/gnu/gcc/gcc-$V/gcc-$V.tar.xz" \
           "https://mirrors.kernel.org/gnu/gcc/gcc-$V/gcc-$V.tar.xz" \
           "https://gcc.gnu.org/pub/gcc/releases/gcc-$V/gcc-$V.tar.xz"; do
      if curl -fsSL --proto '=https' --proto-redir '=https' --connect-timeout 15 --retry 2 "$u" -o "$F"; then break; fi
  done
  echo "${LIBGFORTRAN_LIBQUADMATH_GCC_TARBALL_SHA256:?}  $F" | sha256sum -c - >/dev/null
  echo "OSS bake archives: $(ls "$D"/*.tar.* "$D"/source-builds/*.tar.* 2>/dev/null | wc -l) pinned archives"
}

# Source at the INSTALLED version, so it matches the binary that ships.
bake_apt() {
  D=/usr/share/oss-sources/apt
  mkdir -p "$D"
  sed -i 's/^Types: deb$/Types: deb deb-src/' /etc/apt/sources.list.d/ubuntu.sources
  apt-get update -qq
  # Added = source packages present now, minus the pre-install snapshot.
  dpkg-query -W -f='${source:Package}\t${source:Version}\n' | sort -u \
    | awk -F'\t' 'NR==FNR{base[$0]=1; next} !($1 in base){print $1 "=" $2}' \
          /var/lib/oss-pkgs-before.txt - > /tmp/_added.txt
  cd "$D"
  # Batched; the retry only runs on failure, to name which entry is the problem.
  if ! xargs -a /tmp/_added.txt apt-get source --download-only -q >/dev/null 2>&1; then
      echo "OSS bake apt: batch fetch failed, retrying individually" >&2
      while read -r p; do
          apt-get source --download-only -q "$p" >/dev/null 2>&1 \
              || echo "OSS bake apt: no source package for $p" >&2
      done < /tmp/_added.txt
  fi
  fetched=$(ls ./*.dsc 2>/dev/null | wc -l)
  expected=$(wc -l < /tmp/_added.txt)
  echo "OSS bake apt: ${fetched} source packages added on top of the base image"
  # The retries swallow apt's status, so without this a deb-src outage ships an empty
  # apt/ and still passes. Against the intended count: a partial bake is the same
  # compliance gap as an empty one, only harder to notice.
  if [[ "${fetched}" -lt "${expected}" ]]; then
      echo "OSS bake apt: expected ${expected}; refusing to ship partial source coverage" >&2
      exit 1
  fi
  # deb-src is a build-time need; leaving it on changes `apt-get update` for the user.
  sed -i 's/^Types: deb deb-src$/Types: deb/' /etc/apt/sources.list.d/ubuntu.sources
  rm -rf /var/lib/apt/lists/* /tmp/_added.txt /var/lib/oss-pkgs-before.txt
}

# The sdist for every distribution, uniformly — one rule to state, no per-package judgement about
# whether an installed tree happens to be the source. Where none is published the wheel goes in
# instead; that is not source and is not pretended to be, it is what is being redistributed.
bake_pip() {
  D=/usr/share/oss-sources/pip
  mkdir -p "$D"
  # Tags picking the one wheel that belongs in this image, asked of the interpreter:
  # an empty ${PY_VER} would widen the grep instead of failing.
  PYTAG=$(/opt/venv/bin/python -c 'import sys; print("cp%d%d" % sys.version_info[:2])')
  ARCH=$(uname -m)
  # Already fetched, so a second pass is a delta. Case- and separator-insensitive:
  # sdist filenames keep the project spelling (PyYAML-6.0.3.tar.gz), this loop is PEP 503.
  have=$(ls "$D" 2>/dev/null | tr '[:upper:]_' '[:lower:]-')
  # Excluded: nvidia-*/cuda-* are NVIDIA wheels; paidf-anomalygen and lerobot-stub are
  # this repo. Neither is third-party OSS.
  /opt/venv/bin/python -c '
import importlib.metadata as md, re
for dist in md.distributions():
    name = dist.metadata["Name"]
    if not name:
        continue
    key = re.sub(r"[-_.]+", "-", name).lower()
    if key.startswith(("nvidia-", "cuda-")) or key in ("paidf-anomalygen", "lerobot-stub"):
        continue
    print(key, dist.version)
' | sort -u > /tmp/_dists.txt
  # An empty list would report "0 sdists" and exit 0 — the silent-empty failure again.
  if [[ ! -s /tmp/_dists.txt ]]; then
      echo "OSS bake pip: enumerated no distributions" >&2
      exit 1
  fi
  while read -r name ver; do
      if printf '%s\n' "$have" | grep -q "^${name}-"; then
          continue
      fi
      # One request; the wheel fallback below re-reads this response.
      json=$(curl -fsSL --proto '=https' --proto-redir '=https' "https://pypi.org/pypi/${name}/${ver}/json" 2>/dev/null || true)
      meta=$(printf '%s' "$json" \
          | jq -r '[.urls[] | select(.packagetype=="sdist")]
                   | if length > 0 then (.[0] | [.url, .digests.sha256] | @tsv) else empty end')
      url=$(printf '%s' "$meta" | cut -f1)
      sha=$(printf '%s' "$meta" | cut -f2)
      if [[ -z "$url" ]]; then
          meta=$(printf '%s' "$json" \
              | jq -r '.urls[] | select(.packagetype=="bdist_wheel") | [.url, .digests.sha256] | @tsv' \
              | grep -E "(${PYTAG}|py3-none|py2\.py3-none)" | grep -E "(${ARCH}|any)\.whl" | tail -n1)
          url=$(printf '%s' "$meta" | cut -f1)
          sha=$(printf '%s' "$meta" | cut -f2)
      fi
      if [[ -z "$url" ]]; then
          # Not on PyPI: in this image that means the torch index, a PEP 503 page. Prefer a
          # #sha256 fragment; torch stopped publishing them at 2.13.0, so fall back to the
          # bare link rather than dropping torch out of the bake.
          idx=$(curl -fsSL --proto '=https' --proto-redir '=https' "${TORCH_INDEX_URL}/${name}/" 2>/dev/null || true)
          vpat="-$(printf '%s' "$ver" | sed 's/+/%2B/')-"
          pick() { grep -F -- "$vpat" | grep -E "(${PYTAG}|py3-none)" | grep -E "(${ARCH}|any)" | tail -n1; }
          href=$(printf '%s' "$idx" | grep -oE 'href="[^"]+\.whl#sha256=[a-f0-9]+"' | sed 's/^href="//; s/"$//' | pick)
          if [[ -n "$href" ]]; then
              url=${href%%#*}
              sha=${href##*sha256=}
          else
              url=$(printf '%s' "$idx" | grep -oE 'href="[^"]+\.whl"' | sed 's/^href="//; s/"$//' | pick)
              sha=""
              [[ -n "$url" ]] && echo "OSS bake pip: ${name}==${ver} has no published digest; recorded unverified" >&2
          fi
      fi
      if [[ -z "$url" ]]; then
          # Expected gaps only. Anything else fails the build rather than arriving as one
          # line in a log nobody reads.
          case "$name" in
              apex|flash-attn-3-nv|sam-2) ;;  # source in source-builds/
              cosmos-framework) ;;            # NVIDIA first-party, not third-party OSS
              *) echo "OSS bake pip: unhandled ${name}==${ver} — no sdist, no wheel, no recorded exemption" >&2
                 exit 1 ;;
          esac
          continue
      fi
      # %2B back to +: the index percent-encodes the local version, the artefact does not.
      out="$D/$(basename "${url%%#*}" | sed 's/%2B/+/g')"
      curl -fsSL --proto '=https' --proto-redir '=https' "$url" -o "$out"
      if [[ -n "$sha" ]]; then
          echo "${sha}  ${out}" | sha256sum -c - >/dev/null
      fi
  done < /tmp/_dists.txt
  echo "OSS bake pip: $(ls "$D"/*.tar.* "$D"/*.zip 2>/dev/null | wc -l) sdists, \
$(ls "$D"/*.whl 2>/dev/null | wc -l) wheels (no sdist published)"
  rm -f /tmp/_dists.txt
}

# Excludes its own output, so the file is reproducible from the tree it describes.
bake_manifest() {
  cd /usr/share/oss-sources
  find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256
  echo "OSS bake: $(wc -l < MANIFEST.sha256) files, $(du -sh . | cut -f1) total"
}

case "${1:-}" in
  archives) bake_archives ;;
  apt)      bake_apt ;;
  pip)      bake_pip ;;
  manifest) bake_manifest ;;
  *)        usage ;;
esac
