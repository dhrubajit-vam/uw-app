"""
Time-based train/test split.

We split by Quote_Date rather than randomly: train on the earlier ~80% of
the book, hold out the most recent ~20% as a validation set. This avoids
look-ahead leakage and mirrors how you'd actually validate a pricing model
in production (train on history, test on "the future").
"""
import pandas as pd


def time_based_split(df: pd.DataFrame, date_col: str = "Quote_Date", train_frac: float = 0.8):
    df_sorted = df.sort_values(date_col).reset_index(drop=True)
    cutoff_idx = int(len(df_sorted) * train_frac)
    cutoff_date = df_sorted.loc[cutoff_idx, date_col]
    train = df_sorted[df_sorted[date_col] < cutoff_date].copy()
    test = df_sorted[df_sorted[date_col] >= cutoff_date].copy()
    return train, test, cutoff_date
