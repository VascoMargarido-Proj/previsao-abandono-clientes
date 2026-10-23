import sys
import pandas as pd
from IPython.display import display, Markdown

PATHS = {
    "Accounts": "../data/raw/accounts.csv",
    "Monthly Activity": "../data/raw/monthly_activity.csv",
}

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def md(text):
    display(Markdown(text))


def explore(name):
    """Load one CSV by name, show the report, and return the DataFrame."""
    if name not in PATHS:
        raise KeyError(f"Unknown dataset '{name}'. Choose from: {list(PATHS)}")

    path = PATHS[name]
    df = pd.read_csv(path)

    md(f"## {name}\nFile location: `{path}`")
    md(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")

    md("### Head")
    md(df.head().to_markdown(index=False))

    md("### Column summary")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing %": (df.isna().mean() * 100).round(1),
        "unique": df.nunique(),
    })
    md(summary.to_markdown())

    md("### Describe")
    md(df.describe(include="all").fillna("___").to_markdown())

    return df


if __name__ == "__main__":
    # Lets you run: %run ../src/data_exploration.py Accounts
    for arg in sys.argv[1:]:
        explore(arg)