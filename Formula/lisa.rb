class Lisa < Formula
  desc "lisa-rs: clean-room Rust + Metal LLM inference engine for Qwen3.8-27B"
  homepage "https://github.com/agnosticeng/lisa-rs"
  url "https://github.com/agnosticeng/lisa-rs/releases/download/v0.1.0/lisa-rs-aarch64-darwin.tar.gz"
  version "0.1.0"
  sha256 "b396e18ed420d0cd1045cca06718def4d165503779c3a29923ae84e3e4e49afc"
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
