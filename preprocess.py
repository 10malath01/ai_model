import pandas as pd

def preprocess_data(df):
    df = df.copy()

    # Ensure columns exist
    for col in ["text", "diagnosis", "solution"]:
        if col not in df.columns:
            df[col] = ""

    # Fill missing values
    df["text"] = df["text"].fillna("")
    df["diagnosis"] = df["diagnosis"].fillna("")
    df["solution"] = df["solution"].fillna("")

    # Feature engineering
    df["has_diagnosis"] = df["diagnosis"].apply(lambda x: 1 if len(str(x)) > 10 else 0)
    df["has_solution"] = df["solution"].apply(lambda x: 1 if len(str(x)) > 10 else 0)
    df["text_length"] = df["text"].apply(lambda x: len(str(x)))

    return df
