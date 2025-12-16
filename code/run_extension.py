# ===============================
# Extension: Employment Change
# ===============================

# ---- 1. Imports ----
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# ---- 2. Paths ----
DATA_DIR = Path("../data")
OUT_DIR = Path("../output")
OUT_DIR.mkdir(exist_ok=True)

# ---- 3. Load data ----
df = pd.read_csv(DATA_DIR / "extension.csv")

# ---- 4. Extension outcome: symmetric percent change in FTE ----
df["pchg_fte"] = 2 * (df["fte2"] - df["fte1"]) / (df["fte2"] + df["fte1"])
df.loc[(df["fte2"] + df["fte1"]) == 0, "pchg_fte"] = pd.NA

# ---- 5. Define regression helper ----
def ols_hc1(y, x, data):
    d = data[[y, x]].dropna()
    X = sm.add_constant(d[x])
    return sm.OLS(d[y], X).fit(cov_type="HC1")

# ---- 6. Run extension regressions ----
# NJ dummy
m1 = ols_hc1("pchg_fte", "NJ", df)

# GAP intensity
m2 = ols_hc1("pchg_fte", "GAP", df)

# ---- 7. Save output ----
out = []
out.append("EXTENSION: Change in Headcount Employment (DHC)\n")
out.append("Regression 1: DHC ~ NJ (HC1)\n")
out.append(m1.summary().as_text())
out.append("\n\nRegression 2: DHC ~ GAP (HC1)\n")
out.append(m2.summary().as_text())

(OUT_DIR / "results_extension.txt").write_text("\n".join(out))

print("\n".join(out))
print("\nSaved: output/results_extension.txt")
