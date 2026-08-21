#!/usr/bin/env bash
#
# lisa-rs release script. One run does, via the version in Cargo.toml:
#   1. build + test the binary
#   2. regenerate Formula/lisa.rb (this repo — `brew install agnosticeng/lisa-rs/lisa`)
#   3. commit everything, push to origin
#   4. tag v<version>, push the tag
#   5. gh release create with the binary asset
#
# Requires the GitHub CLI (`gh`) authenticated:
#   brew install gh && gh auth login --git-protocol ssh
#
# Flags:
#   --draft          create a draft release
#   --no-build       skip the cargo build (use existing target/release/cli)
#   --no-tests       skip the fast test suite
#   --no-commit      don't commit/push (build+formula only, no tag/release)
#
set -euo pipefail

SRCDIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SRCDIR"

REPO="${LISA_RS_REPO:-agnosticeng/lisa-rs}"
BIN="cli"
BIN_NAME="lisa"
ARCH="$(uname -m)"; [[ "$ARCH" == arm64 ]] && ARCH="aarch64"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ASSET="lisa-rs-${ARCH}-${OS}.tar.gz"
DRAFT=0
DO_BUILD=1
DO_TESTS=1
DO_COMMIT=1

for arg in "$@"; do
  case "$arg" in
    --draft)       DRAFT=1 ;;
    --no-build)    DO_BUILD=0 ;;
    --no-tests)    DO_TESTS=0 ;;
    --no-commit)   DO_COMMIT=0 ;;
    -h|--help)     grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "error: unknown arg $arg" >&2; exit 2 ;;
  esac
done

# --- gh must be available -----------------------------------------------
command -v gh >/dev/null 2>&1 || { echo "error: gh not installed (brew install gh)" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "error: run 'gh auth login'" >&2; exit 1; }

# --- version from Cargo.toml ---------------------------------------------
VERSION="$(sed -n 's/^version *= *"\([^"]*\)".*/\1/p' Cargo.toml | head -1)"
[[ -n "$VERSION" ]] || { echo "error: could not read version from Cargo.toml" >&2; exit 2; }
TAG="v$VERSION"
echo "==> releasing lisa-rs ${VERSION} (tag ${TAG})"

# --- abort if the tag/release already exists ------------------------------
if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "error: tag ${TAG} already exists locally — bump Cargo.toml or delete the tag" >&2; exit 1
fi
if gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  echo "error: release ${TAG} already exists on ${REPO} — bump Cargo.toml" >&2; exit 1
fi

# --- tests -----------------------------------------------------------------
if [[ "$DO_TESTS" == "1" ]]; then
  echo "==> running cargo test (fast suite)"
  cargo test --release 2>/dev/null | tail -1
fi

# --- build ------------------------------------------------------------------
BINARY="target/release/$BIN"
if [[ "$DO_BUILD" == "1" ]]; then
  echo "==> building ${BIN} (${BIN_NAME}) for ${ARCH}-${OS}"
  cargo build --release --bin "$BIN"
fi
[[ -x "$BINARY" ]] || { echo "error: no built binary at $BINARY (build first)" >&2; exit 1; }

# --- package -----------------------------------------------------------------
echo "==> packaging ${ASSET}"
TMPDIR="$(mktemp -d)"
mkdir -p "$TMPDIR/$BIN_NAME"
cp "$BINARY" "$TMPDIR/$BIN_NAME/lisa"
cp "LICENSE" "$TMPDIR/$BIN_NAME/" 2>/dev/null || true
( cd "$TMPDIR" && tar -czf "$ASSET" "$BIN_NAME" )
ASSET_PATH="$TMPDIR/$ASSET"
sha256sum "$ASSET_PATH" > "$ASSET_PATH.sha256" 2>/dev/null \
  || shasum -a 256 "$ASSET_PATH" > "$ASSET_PATH.sha256"

# --- regenerate the in-repo Homebrew formula ------------------------------
echo "==> regenerating Formula/lisa.rb"
"$(dirname "$0")/gen-formula.sh" "$TAG" "$ASSET_PATH"

# --- commit + push ---------------------------------------------------------
if [[ "$DO_COMMIT" == "1" ]]; then
  echo "==> committing + pushing (${REPO})"
  git add -A
  if git diff --cached --quiet; then
    echo "    nothing to commit"
  else
    git commit -m "lisa-rs ${VERSION}"
    git push origin HEAD
  fi
  echo "==> creating + pushing tag ${TAG}"
  git tag -a "$TAG" -m "lisa-rs ${VERSION}"
  git push origin "$TAG"
fi

# --- release via gh ----------------------------------------------------------
if [[ "$DO_COMMIT" == "1" ]]; then
  NOTES="lisa-rs ${VERSION}

Install: brew install agnosticeng/lisa-rs/lisa
Binary asset: ${ASSET}"

  echo "==> publishing release ${VERSION} (via gh)"
  gh release create "$TAG" \
    "$ASSET_PATH" "$ASSET_PATH.sha256" \
    --repo "$REPO" \
    --title "lisa-rs ${VERSION}" \
    --notes "$NOTES" \
    $( [[ "$DRAFT" == "1" ]] && echo --draft )
  echo "==> done: https://github.com/${REPO}/releases/tag/${TAG}"
else
  echo "==> --no-commit: skipped commit/push/tag/release"
fi

rm -rf "$TMPDIR"
echo "==> done."