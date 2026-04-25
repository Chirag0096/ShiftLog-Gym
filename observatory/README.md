# ShiftLog Observatory

This directory stores evaluation artifacts and a lightweight dashboard for the hackathon demo.

Expected artifacts:

- `episodes.jsonl`
- `memory_events.jsonl`
- `eval_summary.csv`
- `plots/*.png`

Generate the first pass locally with:

```bash
python3 scripts/export_observatory_artifacts.py
```

Then run:

```bash
streamlit run observatory/app.py
```

