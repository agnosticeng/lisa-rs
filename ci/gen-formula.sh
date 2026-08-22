#!/usr/bin/env bash
# Regenerate the Homebrew formula (Formula/lisa.rb) into a target dir for a
# given release tag. The release flow clones the tap repo
# (agnosticeng/homebrew-lisa-rs) and calls this with that clone as the target
# dir, so `brew install agnosticeng/lisa-rs/lisa` resolves without a local
# Formula/ folder in lisa-rs.
# Usage: ./ci/gen-formula.sh v0.1.0 /path/to/lisa-rs-aarch64-darwin.tar.gz [out-dir]
set -euo pipefail
TAG="${1:?usage: gen-formula.sh <tag> <tar.gz-path> [out-dir]}"
TARBALL="${2:?usage: gen-formula.sh <tag> <tar.gz-path> [out-dir]}"
[[ -f "$TARBALL" ]] || { echo "error: not a file: $TARBALL" >&2; exit 1; }
SHA="$(shasum -a 256 "$TARBALL" | awk '{print $1}')"
VER="${TAG#v}"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"; [[ "$(uname -m)" = arm64 ]] && ARCH=aarch64 || ARCH=x86_64
OUT_DIR="${3:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="$OUT_DIR/Formula/lisa.rb"
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