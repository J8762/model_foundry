use clap::Parser;
use serde::{Deserialize, Serialize};
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

/// Command-line interface for the offline model generator / backtester.
#[derive(Parser, Debug)]
#[command(author = "Jay Desk", version = "0.1.0", about = "Rust sweep engine feeding live control plane")]
struct Cli {
    /// Symbol, e.g. XAUUSD
    #[arg(long = "symbol")]
    symbol: String,

    /// Path to tick data CSV: ts_ms,bid,ask
    #[arg(long = "ticks-file")]
    ticks_file: PathBuf,

    /// Optional Donchian length (single-run mode only)
    #[arg(long = "donch-n")]
    donch_n: Option<u32>,

    /// Optional RR min (single-run mode only)
    #[arg(long = "rr-min")]
    rr_min: Option<f64>,

    /// Optional max spread (single-run mode only)
    #[arg(long = "max-spread")]
    max_spread: Option<f64>,

    /// Optional session filter (single-run mode only)
    #[arg(long = "session-filter")]
    session_filter: Option<String>,

    /// If present, run param sweep instead of single-run.
    #[arg(long = "sweep", default_value_t = false)]
    sweep: bool,

    /// Optional JSON file describing the grid for sweep mode.
    /// If provided, overrides built-in grids.
    #[arg(long = "grid-file")]
    grid_file: Option<PathBuf>,
}

// -------- data structures -------- //

#[derive(Debug, Clone, Deserialize)]
struct Tick {
    ts_ms: u64,
    bid: f64,
    ask: f64,
}

#[derive(Debug, Clone)]
struct Candle {
    start_ts: u64,
    end_ts: u64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    mid_spread: f64,
}

#[derive(Debug, Clone, Serialize)]
struct Metrics {
    pf_bt: f64,
    dd_bt: f64,
    trades: u32,
    winrate: f64,
    pnl: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ParamsOut {
    donch_n: u32,
    rr_min: f64,
    max_spread: f64,
    session_filter: String,
}

#[derive(Debug, Clone, Serialize)]
struct CandidateOut {
    model_id: String,
    symbol: String,
    params: ParamsOut,
    metrics: Metrics,
}

#[derive(Debug, Deserialize)]
struct GridSpec {
    donch_n: Vec<u32>,
    rr_min: Vec<f64>,
    session_filter: Vec<String>,
    max_spread: Vec<f64>,
}

// -------- helpers -------- //

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

// Load ticks from CSV with header: ts_ms,bid,ask
fn load_ticks(csv_path: &PathBuf) -> anyhow::Result<Vec<Tick>> {
    let f = File::open(csv_path)?;
    let reader = BufReader::new(f);

    let mut out: Vec<Tick> = Vec::new();
    for (i, line_res) in reader.lines().enumerate() {
        let line = line_res?;
        if i == 0 {
            // assume header row, skip
            if line.contains("ts_ms") && line.contains("bid") && line.contains("ask") {
                continue;
            }
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 3 {
            continue;
        }
        let ts_ms: u64 = parts[0].trim().parse().unwrap_or(0);
        let bid: f64 = parts[1].trim().parse().unwrap_or(0.0);
        let ask: f64 = parts[2].trim().parse().unwrap_or(0.0);

        out.push(Tick { ts_ms, bid, ask });
    }
    Ok(out)
}

// Naive candle builder: bucket ticks into fixed interval_ms windows.
fn ticks_to_candles(ticks: &[Tick], interval_ms: u64) -> Vec<Candle> {
    if ticks.is_empty() {
        return Vec::new();
    }

    let mut candles: Vec<Candle> = Vec::new();

    let mut bucket_start = ticks[0].ts_ms;
    let mut bucket_end = bucket_start + interval_ms;
    let mut cur_open = ticks[0].bid;
    let mut cur_high = ticks[0].bid;
    let mut cur_low = ticks[0].bid;
    let mut cur_close = ticks[0].bid;

    let mut spread_sum = 0.0f64;
    let mut spread_cnt: u32 = 0;

    for tk in ticks {
        if tk.ts_ms >= bucket_end {
            // flush old candle
            let mid_spread = if spread_cnt > 0 {
                spread_sum / (spread_cnt as f64)
            } else {
                0.0
            };
            candles.push(Candle {
                start_ts: bucket_start,
                end_ts: bucket_end,
                open: cur_open,
                high: cur_high,
                low: cur_low,
                close: cur_close,
                mid_spread,
            });

            // advance bucket
            while tk.ts_ms >= bucket_end {
                bucket_start = bucket_end;
                bucket_end = bucket_start + interval_ms;
            }

            // reset state for new candle with current tick
            cur_open = tk.bid;
            cur_high = tk.bid;
            cur_low = tk.bid;
            cur_close = tk.bid;
            spread_sum = (tk.ask - tk.bid).abs();
            spread_cnt = 1;
        } else {
            // same bucket
            if tk.bid > cur_high {
                cur_high = tk.bid;
            }
            if tk.bid < cur_low {
                cur_low = tk.bid;
            }
            cur_close = tk.bid;
            spread_sum += (tk.ask - tk.bid).abs();
            spread_cnt += 1;
        }
    }

    // final candle
    let mid_spread = if spread_cnt > 0 {
        spread_sum / (spread_cnt as f64)
    } else {
        0.0
    };
    candles.push(Candle {
        start_ts: bucket_start,
        end_ts: bucket_end,
        open: cur_open,
        high: cur_high,
        low: cur_low,
        close: cur_close,
        mid_spread,
    });

    candles
}

// Dummy PnL / PF model to keep the shape consistent.
// We derive metrics deterministically from params so
// scoring / ranking is stable without needing real trading logic yet.
fn run_strategy(
    candles: &[Candle],
    donch_n: u32,
    rr_min: f64,
    max_spread: f64,
    session_filter: &str,
) -> Metrics {
    // Trades ~ baseline + function of donch_n
    let trades = 1000 + (donch_n * 5);

    // Winrate ~ 0.50-0.55 range based on rr_min
    let winrate = 0.50 + ((rr_min - 1.5) / 20.0); // rough ~0.50-0.55

    // pf_bt ~ 1.5-1.8 range based on donch_n, rr_min, session
    let session_bonus = match session_filter {
        "london" => 0.05,
        "nyopen" => 0.04,
        "asia" => 0.03,
        _ => 0.02,
    };

    let spread_penalty = if max_spread > 2.5 {
        0.05
    } else if max_spread > 2.0 {
        0.03
    } else {
        0.0
    };

    let pf_bt = 1.5
        + (donch_n as f64) / 200.0
        + (rr_min - 1.5) / 5.0
        + session_bonus
        - spread_penalty;

    // dd_bt ~ 0.04-0.07 range
    let dd_bt = 0.04
        + (donch_n as f64) / 1000.0
        + (rr_min / 100.0)
        + (spread_penalty / 2.0);

    // crude pnl ~ trades * edge
    let pnl = (trades as f64)
        * ((pf_bt - 1.0).max(0.0))
        * (donch_n as f64 / 40.0)
        * 0.01;

    // pretend we're using candles for something, so no warnings
    let _dummy = candles.len();

    Metrics {
        pf_bt,
        dd_bt,
        trades,
        winrate,
        pnl,
    }
}

// Build a candidate struct from params + metrics
fn build_candidate(
    symbol: &str,
    donch_n: u32,
    rr_min: f64,
    max_spread: f64,
    session_filter: &str,
    metrics: Metrics,
) -> CandidateOut {
    let stamp = now_millis();
    let model_id = format!(
        "{}_donch{}_rr{:.2}_{}_{}",
        symbol, donch_n, rr_min, session_filter, stamp
    );

    CandidateOut {
        model_id,
        symbol: symbol.to_string(),
        params: ParamsOut {
            donch_n,
            rr_min,
            max_spread,
            session_filter: session_filter.to_string(),
        },
        metrics,
    }
}

// Score for ranking. Higher is better.
fn score_candidate(c: &CandidateOut) -> f64 {
    // basic: reward PF, punish DD
    c.metrics.pf_bt - (c.metrics.dd_bt * 10.0)
}

// Ensure ./out exists
fn ensure_out_dir() -> anyhow::Result<()> {
    fs::create_dir_all("./out")?;
    Ok(())
}

// Write jsonl lines for all candidates
fn write_candidates_jsonl(cands: &[CandidateOut]) -> anyhow::Result<()> {
    ensure_out_dir()?;
    let mut f = File::create("./out/candidates.jsonl")?;
    for c in cands {
        let line = serde_json::to_string(c)?;
        writeln!(f, "{}", line)?;
    }
    Ok(())
}

// Write top-k to top_candidates.jsonl
fn write_top_candidates_jsonl(cands: &[CandidateOut], top_k: usize) -> anyhow::Result<usize> {
    ensure_out_dir()?;
    let mut v: Vec<CandidateOut> = cands.to_vec();
    // sort by score desc
    v.sort_by(|a, b| score_candidate(b).partial_cmp(&score_candidate(a)).unwrap());

    let kept = std::cmp::min(top_k, v.len());
    let mut f = File::create("./out/top_candidates.jsonl")?;
    for i in 0..kept {
        let line = serde_json::to_string(&v[i])?;
        writeln!(f, "{}", line)?;
    }
    Ok(kept)
}

// Build sweep grid:
// - if grid_file is provided, load GridSpec JSON from disk
// - else, fall back to the built-in mini grid (your current ~18 combos)
fn build_param_grid(
    maybe_grid_file: &Option<PathBuf>,
) -> anyhow::Result<(Vec<u32>, Vec<f64>, Vec<String>, Vec<f64>)> {
    if let Some(grid_path) = maybe_grid_file {
        let raw = fs::read_to_string(grid_path)?;
        let spec: GridSpec = serde_json::from_str(&raw)?;
        return Ok((
            spec.donch_n,
            spec.rr_min,
            spec.session_filter,
            spec.max_spread,
        ));
    }

    // default hard-coded mini grid (what you already proved works)
    let donch_n = vec![20, 30, 40];
    let rr_min = vec![1.8, 2.0, 2.5];
    let session_filter = vec!["london".to_string(), "nyopen".to_string()];
    let max_spread = vec![2.5]; // single spread cap for now

    Ok((donch_n, rr_min, session_filter, max_spread))
}

// -------- main flow -------- //

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // load ticks, build candles (1m candles = 60000 ms)
    let ticks = load_ticks(&cli.ticks_file)?;
    let candles = ticks_to_candles(&ticks, 60_000);

    if cli.sweep {
        // SWEEP MODE
        // Build param grid either from --grid-file or from default
        let (donch_vec, rr_vec, sess_vec, spread_vec) = build_param_grid(&cli.grid_file)?;

        let mut all_candidates: Vec<CandidateOut> = Vec::new();

        for &d in &donch_vec {
            for &rr in &rr_vec {
                for sess in &sess_vec {
                    for &sp in &spread_vec {
                        let metrics = run_strategy(&candles, d, rr, sp, sess);
                        let cand =
                            build_candidate(&cli.symbol, d, rr, sp, sess.as_str(), metrics);
                        all_candidates.push(cand);
                    }
                }
            }
        }

        // write full list
        write_candidates_jsonl(&all_candidates)?;

        // write ranked top 5
        let kept = write_top_candidates_jsonl(&all_candidates, 5)?;

        // print final summary to stdout (this is what your PS expects)
        let summary = serde_json::json!({
            "ok": true,
            "generated": all_candidates.len(),
            "top_kept": kept
        });
        println!("{}", serde_json::to_string(&summary)?);

        return Ok(());
    } else {
        // SINGLE-RUN MODE
        // We expect donch_n, rr_min, max_spread, session_filter to be provided.
        let donch_n = cli
            .donch_n
            .expect("--donch-n is required in single-run mode (no --sweep)");
        let rr_min = cli
            .rr_min
            .expect("--rr-min is required in single-run mode (no --sweep)");
        let max_spread = cli
            .max_spread
            .expect("--max-spread is required in single-run mode (no --sweep)");
        let session_filter = cli.session_filter.clone().expect(
            "--session-filter is required in single-run mode (no --sweep)",
        );

        let metrics = run_strategy(
            &candles,
            donch_n,
            rr_min,
            max_spread,
            session_filter.as_str(),
        );
        let cand = build_candidate(
            &cli.symbol,
            donch_n,
            rr_min,
            max_spread,
            session_filter.as_str(),
            metrics,
        );

        // print one JSON object to stdout
        println!("{}", serde_json::to_string_pretty(&cand)?);
    }

    Ok(())
}
