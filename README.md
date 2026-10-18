# Plot Price Predictor

Interactive four-model regression demo with a FastAPI backend and responsive browser dashboard. Compare Ridge, Random Forest, Gradient Boosting and a trained neural network.

**All training listings and locations are synthetic.** This is not a real property valuation service. Test errors measure held-out synthetic examples only; the displayed error-based range is a heuristic, not a calibrated confidence interval.

## Render

Python 3.12. Build: `pip install -r plot-price-predictor/plot-price-predictor/requirements-render.txt`

Start: `cd plot-price-predictor/plot-price-predictor && uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`

Root URL provides the dashboard; `/docs` documents the API; `/health` confirms models loaded. All four trained artifacts are included. No API key or training step is required.

The inference dependencies match scikit-learn 1.7.2 and NumPy 2 used by the saved artifacts. Neural network weights are exported to JSON and evaluated with NumPy for lower serving memory; original PyTorch weights and training scripts remain available.

See the nested project README for training and experiment details.


## Complete recording workflow
1. Enter a fictional property and run the four trained models.
2. Save current values as a baseline, change area/location/access, and compare the alternative. Show the actual estimate difference and percentage.
3. Upload the sample CSV for batch estimates (up to 100 properties, 1 MB).
4. Reopen a saved run and download the four-model CSV report.

All training listings and locations are synthetic. Estimates are not real market valuations; the displayed range is a MAPE heuristic, not a confidence interval. The backend keeps 20 runs per HTTP-only browser session in SQLite, with temporary storage on free Render. Export results to retain them.
