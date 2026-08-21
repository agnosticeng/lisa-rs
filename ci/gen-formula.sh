#!/usr/bin/env bash
# Regenerate Formula/lisa.rb (in this repo) for a given release tag.
# `brew install agnosticeng/lisa-rs/lisa` resolves to the Formula/ in this
# repo, so the release flow commits the regenerated formula with the release.
# Usage: ./ci/gen-formula.sh v0.1.0 /path/to/lisa-rs-aarch64-darwin.tar.gz
set -euo pipefail
TAG="${1:?usage: gen-formula.sh <tag> <tar.gz-path>}"
TARBALL="${2:?usage: gen-formula.sh <tag> <tar.gz-path>}"
[[ -f "$TARBALL" ]] || { echo "error: not a file: $TARBALL" >&2; exit 1; }
SHA="$(shasum -a 256 "$TARBALL" | awk '{print $1}')"
VER="${TAG#v}"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"; [[ "$(uname -m)" = arm64 ]] && ARCH=aarch64 || ARCH=x86_64
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/Formula/lisa.rb"
mkdir -p "$(dirname "$OUT")"

cat > "$OUT" <<EOF
class Lisa < Formula
  desc "lisa-rs: clean-room Rust + Metal LLM inference engine for Qwen3.8-27B"
  homepage "https://github.com/agnosticeng/lisa-rs"
  url "https://github.com/agnosticeng/lisa-rs/releases/download/${TAG}/lisa-rs-${ARCH}-${OS}.tar.gz"
  version "${VER}"
  sha256 "${SHA}"
  license "MIT"

  depends_on :macos
  depends_on arch: :arm64

  def install
    bin.install Dir["*/lisa", "lisa"][0] => "lisa"
  end

  test do
    assert_match "lisa", shell_output("#{bin}/lisa --help 2>&1", 2)
  end
end
EOF
echo "wrote $OUT (${TAG}, sha256 ${SHA})"