"""Generate a synthetic retail-brokerage dataset (fixed seed => reproducible).

Outputs (data/raw/):
  accounts.csv           one row per account (+ a few duplicate rows)
  monthly_activity.csv   one row per account-month, Jan-2023..Dec-2024
"""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N = 5000
N_MONTHS = 24
rng = np.random.default_rng(SEED)
OUT = Path("data/raw"); OUT.mkdir(parents=True, exist_ok=True)
months = pd.period_range("2023-01", periods=N_MONTHS, freq="M")

# ------------------------------------------------------------------ accounts
acct_id = np.array([f"ACC{i:05d}" for i in range(1, N + 1)])
open_date = pd.Timestamp("2023-01-01") + pd.to_timedelta(
    rng.integers(-36, N_MONTHS - 2, N) * 30.4 + rng.integers(0, 28, N), unit="D")
open_date = pd.Series(open_date).dt.floor("D")
open_month = ((open_date.dt.year - 2023) * 12 + open_date.dt.month - 1).to_numpy()

age = np.clip(rng.normal(41, 12, N), 18, 85).round()
account_type = rng.choice(["Cash", "Margin", "IRA"], N, p=[.55, .25, .20])
channel = rng.choice(["Mobile App", "Web", "Referral", "Advisor Lead"], N, p=[.45, .30, .15, .10])
risk_tol = rng.integers(1, 6, N)
income = np.exp(rng.normal(11.1, 0.5, N)).round(-3)
engagement = rng.gamma(2.0, 1.0, N)
engagement = np.clip(engagement / engagement.mean(), 0.05, None)   # latent trait, never exported
beta = np.clip(rng.normal(1.0 + 0.1 * (risk_tol - 3), 0.25), 0.2, 2.2)
init_bal = np.exp(rng.normal(8.8, 1.1, N))

# ---------------------------------------------------------------- simulation
mkt = rng.normal(0.008, 0.035, N_MONTHS)
mkt[8:13] = rng.normal(-0.035, 0.03, 5)          # bear stretch: Sep-2023 .. Jan-2024

cur = np.zeros(N); started = np.zeros(N, bool)
fade_start = np.full(N, -1)
ret_hist = np.zeros((N_MONTHS, N))
rows = []
for m in range(N_MONTHS):
    live = open_month <= m
    new = live & ~started
    cur[new] = init_bal[new]; started |= new
    k = np.where(fade_start >= 0, m - fade_start, -1)
    mult = np.where(k < 0, 1.0, np.where(k == 0, 0.35, 0.0))   # churners fade, then go silent

    logins = rng.poisson(6.0 * engagement * mult) * live
    trades = rng.poisson(2.2 * engagement * (0.8 + 0.1 * risk_tol) * mult) * live
    ret = beta * mkt[m] + rng.normal(0, 0.02, N)
    ret_hist[m] = ret * live
    dep = (rng.random(N) < np.clip(0.2 * engagement * mult, 0, .9)) * rng.lognormal(6.0, 0.8, N) * live
    wd_p = np.where((k >= 0) & (k <= 1), 0.5, 0.04)             # leavers pull money out
    wd = (rng.random(N) < wd_p) * rng.uniform(0.1, 0.9, N) * cur * live
    fees = trades * 0.65
    cur = np.where(live, np.maximum(cur * (1 + ret) + dep - wd - fees, 0), 0)

    dd = ret_hist[max(0, m - 2): m + 1].sum(axis=0)
    tenure = m - open_month
    fee_ratio = fees / np.maximum(cur, 50)
    logit = (-4.4 - 0.8 * np.log(engagement) + 6.0 * np.clip(-dd, 0, 0.3)
             + 0.7 * (tenure < 6) + 0.6 * (fee_ratio > 0.002)
             + 0.25 * (channel == "Mobile App") - 0.4 * (channel == "Advisor Lead"))
    trig = live & (fade_start < 0) & (rng.random(N) < 1 / (1 + np.exp(-logit)))
    fade_start[trig] = m + 1

    rows.append(pd.DataFrame({
        "account_id": acct_id[live], "m": m, "balance_usd": cur[live].round(2),
        "logins": logins[live], "trades": trades[live],
        "deposits_usd": dep[live].round(2), "withdrawals_usd": wd[live].round(2),
        "fees_usd": fees[live].round(2), "monthly_return_pct": (100 * ret[live]).round(2)}))

act = pd.concat(rows, ignore_index=True)
act["month"] = np.array(months.strftime("%Y-%m"))[act.pop("m").to_numpy()]

# ------------------------------------------------- inject messiness: activity
act = act.drop(act.sample(frac=0.01, random_state=SEED).index).reset_index(drop=True)   # missing months
n = len(act)
act.loc[rng.random(n) < .02, "balance_usd"] = np.nan                                     # missing balances
act.loc[rng.choice(n, 15, replace=False), "balance_usd"] *= 100                          # decimal-shift typos
atype = dict(zip(acct_id, account_type))
cand = act.index[act["account_id"].map(atype).isin(["Cash", "IRA"])]
neg = rng.choice(cand, 20, replace=False)
act.loc[neg, "balance_usd"] = -act.loc[neg, "balance_usd"].abs()                         # impossible negatives
act.loc[rng.random(n) < .005, "logins"] = -1                                             # sentinel for "unknown"
act["trades"] = act["trades"].astype(object)
act.loc[rng.choice(n, 25, replace=False), "trades"] = "n/a"                              # text in numeric col
alt = rng.random(n) < .04
act.loc[alt, "month"] = pd.to_datetime(act.loc[alt, "month"]).dt.strftime("%b-%Y")       # mixed date formats
act = pd.concat([act, act.sample(25, random_state=SEED)], ignore_index=True)             # exact duplicates

# ------------------------------------------------- inject messiness: accounts
fmts = ["%Y-%m-%d", "%m/%d/%Y", "%d %b %Y"]
pick = rng.choice(3, N, p=[.6, .25, .15])
open_str = [d.strftime(fmts[c]) for d, c in zip(open_date, pick)]

variants = {"Mobile App": ["mobile app", "MOBILE APP", "App", "mobile_app"],
            "Web": ["web", "WEB "], "Referral": ["referral", "Ref."],
            "Advisor Lead": ["advisor lead", "Advisor-Lead"]}
channel_raw = [rng.choice(variants[c]) if rng.random() < .35 else c for c in channel]

fmt_mask = rng.random(N) < .30
inc = pd.Series([f"${v:,.0f}" if f else v for v, f in zip(income, fmt_mask)], dtype=object)
inc[rng.random(N) < .12] = np.nan

age[rng.random(N) < .03] = np.nan
age[rng.choice(N, 12, replace=False)] = rng.choice([0, 2, 118, 150, 999], 12)            # impossible ages
risk = risk_tol.astype(float); risk[rng.random(N) < .08] = np.nan

accts = pd.DataFrame({"account_id": acct_id, "open_date": open_str, "age": age,
                      "account_type": account_type, "channel": channel_raw,
                      "risk_tolerance": risk, "stated_income": inc})
accts = pd.concat([accts, accts.sample(40, random_state=SEED)], ignore_index=True)       # duplicate accounts

accts.to_csv(OUT / "accounts.csv", index=False)
act.to_csv(OUT / "monthly_activity.csv", index=False)
print(f"accounts: {accts.shape}, activity: {act.shape}")