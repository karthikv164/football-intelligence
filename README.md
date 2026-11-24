# Football Intelligence — Phase 1

A **read-only**, free-first football betting research prototype. It does **not** replicate OranExcel, execute bets, access private bookmaker handles, or claim to identify professional bettors.

## Setup

Python 3.10+:

```bash
pip install -r requirements.txt
python engine.py --start 2026-11-16 --end 2026-11-22
streamlit run dashboard.py
```

The first run downloads publicly available 2026/27 season CSVs from football-data.co.uk and caches them in `data/`. Network access is needed for the first download. Results are saved under `output/`. To limit scope, use `--leagues E0 D1 SP1 I1 F1`. The site covers selected European divisions, **not all global football leagues**.

## What this test actually measures

For each match with both opening and closing 1X2 bookmaker prices, remove the overround from the three implied probabilities. Pick the outcome with the greatest **opening-to-closing increase** in implied probability if it exceeds a configurable threshold. Evaluate that selection against the actual match result using the quoted closing odds as a hypothetical entry price. Return 1 unit per selection.

**Critical caveat:** This is a retrospective closing-line diagnostic, *not a valid live betting strategy*. Closing odds are not known at opening time; even if they were observed before kickoff, there is no guarantee of execution at the quoted price. Do not interpret the diagnostic ROI as an achievable return. A valid strategy needs timestamped historical odds, defined decision timestamps, held-out testing and real fill/fees assumptions. This is Phase 1 data quality + market signal research only.

## Source & limitations

- Historical match and odds CSVs: https://www.football-data.co.uk/downloadm.php
- Dates depend on publisher updates; if data is incomplete, do not fill in missing results.
- Bookmaker odds are **prices**, not betting volume or wager sizes.
- No whale activity or Kalshi/Polymarket integration is included yet.
- No predictions or profitability guarantees.

## Roadmap

1. Validate actual 2026-11-16 to 2026-11-22 coverage, missingness and source timestamps.
2. Add timestamped odds collector with compliant APIs and durable storage.
3. Add public Kalshi/Polymarket trade adapters; separate matched volume, available liquidity, and market odds.
4. Map markets to canonical fixture IDs; resolve timezone/team aliases.
5. Add pre-kickoff team news/lineups with timestamped sources.
6. Train a leakage-free predictive model and test on genuinely unseen fixtures.
7. Extend to licensed international odds feeds only when cost and accuracy justify it.
