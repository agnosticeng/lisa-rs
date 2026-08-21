// mactop-style terminal dashboard for the OpenAI-compatible server.
//
// Renders live gauges (prefill speed, token speed, speculative acceptance,
// draft ratio) plus a query list of completed/in-flight requests, like a
// lightweight htop/mactop for the inference server. Run on the main thread;
// the axum server runs in a background thread.
use std::io;
use std::time::Duration;

use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table};

use super::metrics::{Metrics, QueryRecord, SharedMetrics};

/// Keyboard input event (decoupled from crossterm so tests can drive the UI).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum UiEvent {
    Quit,
    Reset,
    Up,
    Down,
    None,
}

#[derive(Default)]
struct UiState {
    max_prefill_tok_s: f64,
    max_decode_tok_s: f64,
    scroll: usize,
}

pub fn run_dashboard(metrics: &SharedMetrics, model_id: String, mtp_id: Option<String>) -> io::Result<()> {
    let mut terminal = ratatui::init();
    let mut state = UiState::default();
    let res = run_loop(&mut terminal, metrics, &model_id, mtp_id.as_deref(), &mut state);
    ratatui::restore();
    res
}

fn run_loop(
    terminal: &mut ratatui::DefaultTerminal,
    metrics: &SharedMetrics,
    model_id: &str,
    mtp_id: Option<&str>,
    state: &mut UiState,
) -> io::Result<()> {
    loop {
        terminal.draw(|frame| draw(frame, metrics, model_id, mtp_id, state))?;
        match poll_event()? {
            UiEvent::Quit => return Ok(()),
            UiEvent::Reset => {
                if let Ok(mut m) = metrics.lock() {
                    *m = Metrics::new();
                }
            }
            UiEvent::Up => state.scroll = state.scroll.saturating_add(1),
            UiEvent::Down => state.scroll = state.scroll.saturating_sub(1),
            UiEvent::None => {}
        }
    }
}

fn poll_event() -> io::Result<UiEvent> {
    use crossterm::event::{Event, KeyCode, KeyEventKind};
    if !crossterm::event::poll(Duration::from_millis(100))? {
        return Ok(UiEvent::None);
    }
    match crossterm::event::read()? {
        Event::Key(key) if key.kind == KeyEventKind::Press => match key.code {
            KeyCode::Char('q') | KeyCode::Esc => Ok(UiEvent::Quit),
            KeyCode::Char('r') => Ok(UiEvent::Reset),
            KeyCode::Up | KeyCode::Char('k') => Ok(UiEvent::Up),
            KeyCode::Down | KeyCode::Char('j') => Ok(UiEvent::Down),
            _ => Ok(UiEvent::None),
        },
        _ => Ok(UiEvent::None),
    }
}

// ---------------------------------------------------------------------------
// draw
// ---------------------------------------------------------------------------

fn draw(
    frame: &mut Frame,
    metrics: &SharedMetrics,
    model_id: &str,
    mtp_id: Option<&str>,
    state: &mut UiState,
) {
    let m = metrics.lock().unwrap();
    update_maxes(&m, state);

    let area = frame.area();
    let [header_area, _, stats_area, _, summary_area, _, table_area, footer_area] = Layout::vertical([
        Constraint::Length(1),
        Constraint::Length(1),
        Constraint::Length(3),
        Constraint::Length(1),
        Constraint::Length(3),
        Constraint::Length(1),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .areas(area);

    // --- header -----------------------------------------------------------
    let mut header_spans = vec![
        Span::styled(
            "lisa-rs",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(model_id, Style::default().fg(Color::DarkGray)),
    ];
    if mtp_id.is_some() {
        header_spans.push(Span::raw("  "));
        header_spans.push(
            Span::styled("MTP", Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)),
        );
    }
    frame.render_widget(Paragraph::new(Line::from(header_spans)), header_area);

    // --- stats panel -------------------------------------------------------
    let (_, prefill_label, _) = gauge_data(&m, state, |r| r.prefill_tok_s, state.max_prefill_tok_s);
    let (_, decode_label, _) = gauge_data(&m, state, |r| r.decode_tok_s, state.max_decode_tok_s);
    let (_, accept_label, _) = ratio_data(&m, |r| r.acceptance);
    let (_, draft_label, _) = ratio_data(&m, |r| r.draft_ratio);

    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("Prefill ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{prefill_label} tok/s"), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::raw("  |  "),
            Span::styled("Decode ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{decode_label} tok/s"), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::raw("  |  "),
            Span::styled("Accept ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{accept_label}%"), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::raw("  |  "),
            Span::styled("Draft ", Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{draft_label}%"), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        ]))
        .block(panel_block("Stats")),
        stats_area,
    );

    // --- summary strip (mactop Power/Silicon/Network style) ---------------
    let summary_cols = Layout::horizontal([
        Constraint::Percentage(40),
        Constraint::Percentage(30),
        Constraint::Percentage(30),
    ])
    .split(summary_area);

    // Totals panel
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                format!(
                    "Tokens: {} gen / {} draft / {:.1}% acc",
                    m.aggregates.total_completion_tokens,
                    m.aggregates.total_drafted_tokens,
                    percent(m.aggregates.total_accepted_drafts, m.aggregates.total_drafted_tokens),
                ),
                Style::default().fg(Color::White),
            ),
        ]))
        .block(panel_block("Totals")),
        summary_cols[0],
    );

    // Timing panel
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                format!(
                    "Prefill {:.0}ms | Decode {:.0}ms",
                    m.aggregates.prefill_ms, m.aggregates.decode_ms
                ),
                Style::default().fg(Color::White),
            ),
        ]))
        .block(panel_block("Timing")),
        summary_cols[1],
    );

    // Throughput panel
    let total_tok = m.aggregates.total_completion_tokens;
    let total_s = (m.aggregates.decode_ms + m.aggregates.prefill_ms) / 1000.0;
    let avg_tok_s = if total_s > 0.0 {
        total_tok as f64 / total_s
    } else {
        0.0
    };
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                format!("Avg: {avg_tok_s:.1} tok/s | {total_tok} tokens total"),
                Style::default().fg(Color::White),
            ),
        ]))
        .block(panel_block("Throughput")),
        summary_cols[2],
    );

    // --- query table ------------------------------------------------------
    frame.render_widget(query_table(&m.queries, state.scroll), table_area);

    // --- footer (mactop style) --------------------------------------------
    let footer = Paragraph::new(Line::from(vec![
        Span::raw(" "),
        Span::styled("q", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        Span::raw(" quit  "),
        Span::styled("r", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        Span::raw(" reset  "),
        Span::styled("\u{2191}/\u{2193}", Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
        Span::raw(" scroll"),
    ]));
    frame.render_widget(footer, footer_area);
}

// ---------------------------------------------------------------------------
// panel block
// ---------------------------------------------------------------------------

fn panel_block(title: &str) -> Block<'static> {
    Block::bordered()
        .borders(Borders::ALL)
        .title(Line::from(title.to_string()).bold())
        .border_style(Style::default().fg(Color::White))
}

// ---------------------------------------------------------------------------
// gauge helpers
// ---------------------------------------------------------------------------

fn update_maxes(m: &Metrics, state: &mut UiState) {
    for r in m.queries.iter().chain(m.active.iter()) {
        state.max_prefill_tok_s = state.max_prefill_tok_s.max(r.prefill_tok_s);
        state.max_decode_tok_s = state.max_decode_tok_s.max(r.decode_tok_s);
    }
    state.max_prefill_tok_s = state.max_prefill_tok_s.max(1.0);
    state.max_decode_tok_s = state.max_decode_tok_s.max(1.0);
}

fn gauge_data(
    m: &Metrics,
    _state: &UiState,
    f: impl Fn(&QueryRecord) -> f64,
    max: f64,
) -> (f64, String, u16) {
    let (val, label) = latest_speed(m, f);
    let pct = ratio_100(val, max);
    (val, label, pct)
}

fn ratio_data(m: &Metrics, f: impl Fn(&QueryRecord) -> f64) -> (f64, String, u16) {
    let (pct, raw) = latest_ratio(m, f);
    (raw, format!("{raw:.1}"), pct)
}

fn latest_speed(m: &Metrics, f: impl Fn(&QueryRecord) -> f64) -> (f64, String) {
    let src: Vec<&QueryRecord> = m.active.iter().chain(m.queries.iter()).collect();
    for r in &src {
        if f(r) > 0.0 {
            let v = f(r);
            return (v, format!("{v:.0}"));
        }
    }
    (0.0, "0".into())
}

fn latest_ratio(m: &Metrics, f: impl Fn(&QueryRecord) -> f64) -> (u16, f64) {
    let src: Vec<&QueryRecord> = m.active.iter().chain(m.queries.iter()).collect();
    let v = src
        .iter()
        .map(|r| f(r) * 100.0)
        .find(|v| *v >= f32::EPSILON as f64 * 100.0)
        .unwrap_or(0.0);
    ((v.clamp(0.0, 100.0)) as u16, v)
}

fn percent(num: u64, den: u64) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64 * 100.0
    }
}

fn ratio_100(value: f64, max: f64) -> u16 {
    if max <= 0.0 {
        return 0;
    }
    ((value / max).clamp(0.0, 1.0) * 100.0) as u16
}

// ---------------------------------------------------------------------------
// query table
// ---------------------------------------------------------------------------

fn query_table(queries: &[QueryRecord], scroll: usize) -> Table<'static> {
    let header = Row::new(vec![
        "#", "TIME", "PROMPT", "PT", "CT", "PREF ms", "PREF t/s", "DEC t/s", "ACC%", "DRFT", "FWD",
    ])
    .style(
        Style::default()
            .fg(Color::White)
            .add_modifier(Modifier::BOLD),
    )
    .bottom_margin(0);

    let max_skip = queries.len().saturating_sub(1);
    let rows: Vec<Row> = queries
        .iter()
        .enumerate()
        .filter(|(i, _)| *i >= scroll.min(max_skip))
        .map(|(_, q)| {
            let time = format_time(q.started_unix_ms);
            let status = if q.done { "" } else { "\u{25f7} " };
            Row::new(vec![
                format!("#{}", q.seq),
                time,
                format!("{status}{}", q.prompt),
                q.prompt_tokens.to_string(),
                q.completion_tokens.to_string(),
                format!("{:.0}", q.prefill_ms),
                format!("{:.0}", q.prefill_tok_s),
                format!("{:.0}", q.decode_tok_s),
                format!("{:.0}", q.acceptance * 100.0),
                format!("{:.0}%", (q.draft_ratio * 100.0).min(999.0)),
                q.target_forwards.to_string(),
            ])
        })
        .collect();

    Table::new(
        rows,
        [
            Constraint::Length(5),
            Constraint::Length(5),
            Constraint::Min(20),
            Constraint::Length(4),
            Constraint::Length(4),
            Constraint::Length(7),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(6),
            Constraint::Length(6),
            Constraint::Length(5),
        ],
    )
    .header(header)
    .block(
        Block::bordered()
            .borders(Borders::ALL)
            .title(Line::from("Queries".to_string()).bold())
            .border_style(Style::default().fg(Color::White)),
    )
    .column_spacing(1)
    .style(Style::default())
}

fn format_time(unix_ms: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let secs = unix_ms / 1000;
    let ago = now.saturating_sub(secs);
    format!("{ago}s")
}

/// Render one frame into an in-memory buffer and print it as plain text.
pub fn preview_draw(metrics: &SharedMetrics, model_id: &str, mtp_id: Option<&str>) {
    let mut terminal =
        ratatui::Terminal::new(ratatui::backend::TestBackend::new(140, 40)).unwrap();
    let mut state = UiState::default();
    terminal
        .draw(|frame| draw(frame, metrics, model_id, mtp_id, &mut state))
        .unwrap();
    let buffer = terminal.backend_mut().buffer();
    let width = buffer.area.width as usize;
    for chunk in buffer.content.chunks(width) {
        let line: String = chunk.iter().map(|c| c.symbol()).collect::<String>();
        println!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::metrics::Metrics;

    fn render(metrics: &SharedMetrics, model_id: &str) {
        let backend = ratatui::backend::TestBackend::new(120, 40);
        let mut terminal = ratatui::Terminal::new(backend.clone()).unwrap();
        let mut state = UiState::default();
        terminal
            .draw(|frame| draw(frame, metrics, model_id, None, &mut state))
            .unwrap();
        backend.buffer();
    }

    #[test]
    fn renders_empty_dashboard() {
        let metrics = Metrics::shared();
        render(&metrics, "qwen3.8-27b");
    }

    #[test]
    fn renders_populated_dashboard() {
        let metrics = Metrics::shared();
        {
            let mut m = metrics.lock().unwrap();
            for i in 0..3 {
                m.begin(format!("What is the capital of country #{i}?"), 55);
                m.tick(20 + i);
                m.finish(350.0, 900.0, 18, 40, 25);
            }
            m.begin("And also this longer question in flight...".to_string(), 80);
            m.tick(10);
        }
        render(&metrics, "qwen3.8-27b");
    }

    #[test]
    fn metrics_aggregate_correctly() {
        let metrics = Metrics::shared();
        let mut m = metrics.lock().unwrap();
        m.begin("prompt a".to_string(), 10);
        m.tick(5);
        m.finish(100.0, 500.0, 4, 8, 6);
        m.begin("prompt b".to_string(), 20);
        m.tick(3);
        m.finish(50.0, 300.0, 2, 4, 2);

        let agg = m.aggregates;
        assert_eq!(agg.total_queries, 2);
        assert_eq!(agg.total_completion_tokens, 8);
        assert_eq!(agg.total_prompt_tokens, 30);
        assert_eq!(agg.total_drafted_tokens, 12);
        assert_eq!(agg.total_accepted_drafts, 8);
        assert!((agg.prefill_ms - 150.0).abs() < 1e-9);
        assert!((agg.decode_ms - 800.0).abs() < 1e-9);

        let first = m.queries.first().unwrap();
        assert_eq!(first.prompt, "prompt b");
        assert_eq!(first.completion_tokens, 3);
        assert!((first.acceptance - 0.5).abs() < 1e-9);
        assert!((first.draft_ratio - 4.0 / 3.0).abs() < 1e-9);
        assert!((first.decode_tok_s - 10.0).abs() < 1e-6);
    }

    #[test]
    fn reset_clears_metrics() {
        let metrics = Metrics::shared();
        {
            let mut m = metrics.lock().unwrap();
            m.begin("x".to_string(), 5);
            m.tick(2);
            m.finish(10.0, 20.0, 1, 2, 1);
        }
        {
            let mut m = metrics.lock().unwrap();
            assert_eq!(m.aggregates.total_queries, 1);
            *m = Metrics::new();
            assert_eq!(m.aggregates.total_queries, 0);
        }
    }
}
