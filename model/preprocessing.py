"""
Shared data-cleaning and preprocessing utilities for the UNSW-NB15
network intrusion detection dataset.

Used by both:
  - model/train_models.py   (offline training)
  - app.py                  (Streamlit app, at inference time)

Keeping this logic in one place guarantees that any CSV a user uploads to
the Streamlit app is cleaned in EXACTLY the same way the training data was
cleaned, which is what makes the saved model pipelines usable on new data.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "attack_cat"

# Columns that must never be used as model inputs:
#   - id     : row identifier, no predictive meaning
#   - label  : binary "is this an attack?" flag. It is derived directly from
#              attack_cat (label = 0 only when attack_cat == "Normal"), so
#              using it as a feature would leak the answer to the model.
DROP_COLS = ["id", "label"]

CATEGORICAL_COLS = ["proto", "service", "state"]

# The raw `proto` field has 133 distinct values but a handful of protocols
# dominate the traffic. We bucket the long tail into "other" so one-hot
# encoding doesn't blow up into 130+ near-empty columns.
TOP_PROTOCOLS = ["tcp", "udp", "unas", "arp", "ospf", "sctp"]

# The full raw feature set expected in an uploaded CSV (order does not matter).
RAW_FEATURE_COLUMNS = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
    "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]

ATTACK_CATEGORIES = [
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
    "Analysis", "Backdoor", "Shellcode", "Worms",
]


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning applied identically at train and inference time."""
    df = df.copy()

    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Normalize text fields (some exports pad values with whitespace).
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "proto" in df.columns:
        df["proto"] = df["proto"].where(df["proto"].isin(TOP_PROTOCOLS), "other")

    if "attack_cat" in df.columns:
        df["attack_cat"] = df["attack_cat"].astype(str).str.strip()

    # Any stray missing numeric values (there are none in the source data,
    # but this keeps the app robust against edited/uploaded CSVs).
    numeric_cols = [c for c in RAW_FEATURE_COLUMNS if c not in CATEGORICAL_COLS]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    return df


def split_features_target(df: pd.DataFrame):
    """Returns (X, y). y is None if the CSV has no attack_cat column."""
    df = df.copy()
    y = None
    if TARGET_COL in df.columns:
        y = df[TARGET_COL]
        df = df.drop(columns=[TARGET_COL])

    missing = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing expected columns: {missing}")

    X = df[RAW_FEATURE_COLUMNS]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            ("num", StandardScaler(), numeric_cols),
        ]
    )
    return preprocessor
