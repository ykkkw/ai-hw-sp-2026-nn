# Hourly Energy Consumption — RNN vs LSTM Forecasting

A PyTorch study comparing Simple RNN and LSTM architectures for demand forecasting on the DOM hourly dataset.

DOM_hourly.csv from the [PJM Hourly Energy Consumption dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption/data?select=DOM_hourly.csv) on Kaggle.

---

## Models

Both models are defined in `models.py` and share the same architecture pattern: a stacked recurrent layer → dropout → linear projection to a single output.

**Shared hyperparameters:**

| Param | Value |
|-------|-------|
| hidden_size | 64 |
| num_layers | 3 |
| dropout | 0.15 |
| optimizer | Adam |
| learning rate | 1e-3 |
| epochs | 30 |
| batch_size | 512 |
| sequence_length | 24 |

---

## Notebook Walkthrough

### Part I — EDA
Visualizes the full time series, demand distributions, and hourly/daily/monthly seasonal patterns.

### Part II — Preprocessing
- **Outlier detection:** IQR fencing (clip to `[Q1 − 1.5×IQR, Q3 + 1.5×IQR]`)
- **Train/val/test split** (time-split):

| Split | Proportion | Records | Period |
|-------|-----------|---------|--------|
| Train | 70% | ~81k | 2005–2014 |
| Validation | 15% | ~17k | 2014–2016 |
| Test | 15% | ~17k | 2016–2018 |

- **Normalization:** MinMaxScaler fitted on train only, then applied to val/test to prevent data leakage.

### Part III — Feature Engineering
Six cyclical time embeddings encode hour-of-day, day-of-week, and month as sin/cos pairs, avoiding the ordinal discontinuity problem (e.g., hour 23 and hour 0 are treated as adjacent).

```
sin(2π·h/24),  cos(2π·h/24)   # hour of day
sin(2π·d/7),   cos(2π·d/7)    # day of week
sin(2π·m/12),  cos(2π·m/12)   # month
```

### Part IV — Dataset & DataLoader
A sliding-window `Dataset` produces `(seq_len=24, input_features)` → `(1,)` sample pairs. Input will be in lag array format with length 24.

### Part V — Training & Evaluation
Trained four models RNN base, RNN + feature, LSTM base, and LSTM+feature. Predictions are evaluated on RMSE, MAE, R², and MAPE.
- RNN total parameters: 20,993
- LSTM total parameters: 83,777 (~4 times RNN params since it has 4 gates)

---

## Key Findings

- **LSTM outperforms Simple RNN** across all metrics — the gated cell state retains relevant information over 24+ timestep sequences, where vanilla RNN vanishing gradients cause degraded performance.
- **Cyclical time features improve both models** — explicitly encoding hour, weekday, and season offloads periodic pattern learning from the recurrent state.
- **Outlier clipping helps** — removing extreme values before scaling leads to more stable training.

## Links
Youtube link for walk through code: https://www.youtube.com/watch?v=ttesidRzbk0


## References
- [Kaggle data](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption/data?select=DOM_hourly.csv)
- [Cyclical feature encoding](https://www.kaggle.com/code/avanwyk/encoding-cyclical-features-for-deep-learning)
- [PyTorch nn.RNN docs](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
- [PyTorch nn.LSTM docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [Sequence length](https://stackoverflow.com/questions/49573242/what-is-sequence-length-in-lstm)
- [MinMaxScaler — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)
- [R² score — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
