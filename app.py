import pandas as pd
import re
from pathlib import Path

# -------- CONFIG --------
INPUT = r"C:\Users\DINESH\PycharmProjects\MiniProject\emails.csv"    # raw Kaggle file
OUTPUT = "emails_parsed.csv" # parsed output file

def parse_email(raw):
    if pd.isna(raw):
        return pd.Series([None]*9)

    raw = str(raw)

    # 1) Split header vs body
    parts = re.split(r"\r?\n\r?\n", raw, maxsplit=1)
    header = parts[0]
    body = parts[1] if len(parts) > 1 else ""

    # 2) Regex to find header fields
    def find(pattern):
        m = re.search(pattern, header, flags=re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None

    from_    = find(r"^From:\s*(.*)")
    to_      = find(r"^To:\s*(.*)")
    date_    = find(r"^Date:\s*(.*)")
    subject_ = find(r"^Subject:\s*(.*)")

    # 3) Tokenization
    body_tokens = re.findall(r"\b\w+\b", body.lower()) if body else []
    body_token_count = len(body_tokens)

    subject_tokens = re.findall(r"\b\w+\b", subject_.lower()) if subject_ else []
    subject_token_count = len(subject_tokens)

    return pd.Series([
        from_,
        to_,
        date_,
        subject_,
        body,
        body_tokens,
        body_token_count,
        subject_tokens,
        subject_token_count
    ])

def main():
    input_path = Path(INPUT)
    if not input_path.exists():
        print("❌ File not found:", INPUT)
        return

    print("Reading dataset...")
    df = pd.read_csv(INPUT, encoding="latin1", low_memory=False)

    # Use column that contains raw emails
    raw_col = "message" if "message" in df.columns else df.columns[0]
    print("Using column:", raw_col)

    # Apply parser
    parsed = df[raw_col].apply(parse_email)
    parsed.columns = [
        "From",
        "To",
        "Date",
        "Subject",
        "Body",
        "BodyTokens",
        "BodyTokenCount",
        "SubjectTokens",
        "SubjectTokenCount"
    ]

    # Join with other columns
    final = pd.concat([df.drop(columns=[raw_col]), parsed], axis=1)

    # Save new CSV
    final.to_csv(OUTPUT, index=False, encoding="utf-8")
    print("✅ Done. Saved as", OUTPUT)

if __name__ == "_main_":
    main()
