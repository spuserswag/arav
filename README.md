# BTC Up/Down Classifier

A machine-learning project that tries to predict whether Bitcoin's price will
be **higher or lower 15 seconds from now**, using live order-book data (the
queue of buy and sell offers on Binance).

This README is written for someone who hasn't worked with transformers before.
If you've used logistic regression or random forests, you're in good shape;
the transformer is just a fancier version of the same idea (take a bunch of
features, predict a label).

---

## The 60-second pitch

1. **Input:** A snapshot of Binance's order book every ~0.25 seconds (bids,
   asks, spread, a few imbalance ratios). We collect these as CSVs.
2. **Features:** We turn each snapshot into a row of numbers — prices,
   spreads, volatility estimates over different time windows, short-term
   returns, etc.
3. **Label:** For each row, we look 15 seconds into the future and ask:
   did the mid-price go up or down? 1 = up, 0 = down.
4. **Model:** A small transformer reads the last 30 rows (15 seconds of
   history) and outputs a probability the price will go up.
5. **Output:** During training we save a checkpoint, some diagnostic plots,
   and a one-page metrics summary.

---

## What's a transformer, briefly

A transformer is a neural network that's good at looking at a *sequence* of
inputs and figuring out which parts of the sequence matter most for the
prediction. It was originally invented for language ("what word comes next
given this sentence?"), but it works for time series too ("what happens
next given this 15-second history?").

The core idea is **attention**: for each timestep, the model computes a
weighted average of *all other timesteps* in the window, where the weights
are learned. So if a spike at t-5 seconds is a strong signal, the model can
"pay attention" to it when predicting at time t.

What you actually need to know to work on this codebase:

- **Input** is a tensor of shape `(batch, seq_len, num_features)` — here,
  `(B, 30, 14)`. That's 30 half-second grid points × 14 features per point.
- **Output** is a single number per sample: the log-odds (logit) that the
  next 15 seconds will be "up". We convert to probability with a sigmoid.
- The model has ~**hundreds of thousands** of parameters (not billions).
  It's tiny. We can train on a laptop GPU in minutes.

---

## Repository layout

```
.
├── data/part2/                   # raw CSVs go here (gitignored)
├── dataloader.py                 # CSV → features + labels; the boring-but-critical part
├── transformer.py                # the model + the training loop
├── tsne.py                       # make a 2D plot of the model's internal representations
├── XGBoost_base_features.py      # a simple baseline we compare against
├── initial.py                    # even simpler baselines (logistic regression, random forest)
├── live_inference.py             # run a trained model against live data
├── sweep.py                      # try many hyperparameter combos, pick the best
├── transformer.ipynb             # notebook version of the above (optional)
├── v1/                           # frozen snapshot of the original pipeline
└── outputs/                      # training runs land here (gitignored)
```

**Which file should you start with?** `dataloader.py`. That's where the
domain knowledge lives. Read the top of it — the docstring explains every
design decision. If you understand how a row becomes a feature vector and
how the label is constructed, you understand 80% of this project.

After that, skim `transformer.py`'s `train_transformer()` function. That's
the training loop. Everything above it is model-definition detail you can
treat as a black box until you need to touch it.

---

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd <repo>

# 2. Python environment (3.10+ recommended)
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install deps
pip install torch numpy pandas scikit-learn matplotlib seaborn xgboost ccxt

# 4. Put a CSV in data/part2/
mkdir -p data/part2
# Drop advanced_orderflow_ws.csv (or your own capture) in there.
# It's gitignored — Spenser will share it separately.
```

Everything else is handled by scripts.

---

## Running things

All scripts can be run from the repo root. They all work with reasonable defaults.

**Quick baselines (sanity-check the data, ~1 minute):**
```bash
python initial.py
python XGBoost_base_features.py
```
XGBoost should hit a validation AUC somewhere in the 0.55–0.65 range on this
data. If it doesn't, something is wrong with the CSV, not the model.

**Train the transformer:**
```bash
python transformer.py --epochs 25 --patience 5 --seed 42
```
This takes ~5–15 minutes on a GPU, longer on CPU. When it finishes it writes
everything to `outputs/v<N>/` where `N` is one more than the previous run.
Open `training_curves.png` — that shows loss and AUC over time and is the
fastest way to tell if training went well.

**Hyperparameter sweep:**
```bash
python sweep.py --preset regularize --epochs-per-run 25
```
Tries 27 combinations of dropout / weight-decay / attention-diagonal-bias,
writes a leaderboard CSV sorted by validation AUC. Useful for "is there a
better hyperparameter?" but expensive (an hour or more).

---

## Interpreting an output folder

Every `outputs/v<N>/` has:

- `transformer_best.pt` — the trained model weights (the checkpoint with
  the best validation AUC we saw during training).
- `scaler.pkl` — the feature scaler fit on the training data. You need
  this at inference time to transform new data the same way training saw it.
- `config.json` — the full run configuration (data knobs, model knobs,
  training knobs) plus results. Good for reproducibility and for
  comparing runs.
- `metrics_summary.txt` — a one-page "how did this run do" report.
- `training_curves.png` — loss and AUC per epoch.
- `attention_heatmap.png` — what parts of the window the model attends to.
- `tsne.png` — a 2D scatter of the model's internal embeddings, colored
  by true label and by predicted probability. Clustering by color = the
  model has learned something meaningful.

---

## How to tell if a run "worked"

Look at `metrics_summary.txt`:

- **Test AUC > 0.55** means the model has *some* edge. Financial time-series
  AUC above ~0.60 is hard to come by; don't expect 0.9.
- **Train AUC >> val AUC** (like 0.99 vs 0.55) means the model memorized
  training data but didn't generalize. Fix: more regularization (higher
  `--dropout`, higher `--weight-decay`, smaller `--d-model`).
- **Val loss going up while train loss goes down** used to be our main bug
  and was caused by non-stationary features (raw BTC price leaked the
  train/val split into the model). That's fixed now in `dataloader.py` —
  but if you change features, be careful not to reintroduce it.

---

## Known ongoing issues

- **Signal is weak.** At a 15-second horizon the true "up vs down" rate
  given the features is probably 52–58% at best. That's realistic for
  high-frequency price prediction; don't expect anything like 70%.
- **Market regime shifts.** Our train set was from a bid-heavy period;
  val/test was ask-heavy. The model's decision boundary gets invalidated
  by regime changes. Possible fixes live in `dataloader.py` (try
  demeaning `obi_10` per-window).
- **Small effective dataset after cleanup.** 407k raw ticks become ~35k
  half-second grid points after resample/dropna/dead-band. Val/test are
  only ~5k rows each. More data (longer capture) would help.

---

## Good first tasks for a new contributor

Pick any of these — they're all useful and don't require deep transformer
knowledge:

1. **Extend the feature set.** Trade volume, order cancellation rate, and
   top-of-book update frequency would all be new signal. Add them in
   `dataloader.py`, rerun, compare.
2. **Per-window normalization.** Instead of a global `StandardScaler`,
   z-score each 30-step window against its own mean/std. This would
   eliminate the regime-shift problem entirely. Implement as a model
   input transform.
3. **Run the sweep on more horizons.** Our 15-second target may be too
   short. `python sweep.py --preset horizon` tries 5s, 10s, 15s, 30s, 60s.
4. **Compare XGBoost and the transformer fairly.** They currently use the
   same feature set but different train/val splits. Unify the eval code
   so we can report head-to-head on the exact same test samples.
5. **Write a proper backtest.** We have `live_inference.py` for
   forward-looking inference, but no historical PnL backtest. A simple
   "long when p > threshold, flat otherwise" simulation on the test set
   would tell us whether the AUC edge translates to money.

---

## Questions

Ping Spenser on the usual channel. The most productive question format is
"I ran X, I expected Y, I got Z, here's the log file" — easy to help with.
# arav
