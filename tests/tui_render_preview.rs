//! Ignored helper: renders the dashboard into a TestBackend buffer and prints
//! it as text, for eyeballing the mactop-style layout without a real terminal.
use lisa_rs::serve::metrics::Metrics;
use lisa_rs::serve::ui::preview_draw;

#[test]
#[ignore = "manual preview: cargo test --test tui_render_preview -- --ignored"]
fn preview() {
    let metrics = Metrics::shared();
    {
        let mut m = metrics.lock().unwrap();
        for i in 0..3 {
            m.begin(format!("What is the capital of country #{i} and its largest city? Provide details."), 61);
            m.tick(28 + i);
            m.finish(212.0, 912.0, 18, 42, 27);
        }
        m.begin("Explain the difference between stable merge sort and in-place.".to_string(), 74);
        m.tick(15);
    }
    preview_draw(&metrics, "qwen3.8-27b", Some("qwen3.8-27b-mtp"));
}
