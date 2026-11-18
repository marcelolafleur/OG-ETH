"""
Credit and saving in the Ethiopia Socioeconomic Panel Survey 2021/22 (Wave 5),
the tabulations quoted in the household calibration chapter.

The public-use microdata (Stata files) come from the World Bank Microdata
Library, catalog 6161, after registration; they may not be redistributed, so
they are not in this repository. Unzip them anywhere and run

    python examples/ess_credit_saving.py <unzipped folder> <output folder>

"""

import sys
import glob
from pathlib import Path
import pandas as pd

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])


def rd(name):
    return pd.read_stata(glob.glob(str(R / "**" / name), recursive=True)[0])


cons = rd("cons_agg_w5.dta")[["household_id", "cons_quint", "pw_w5", "saq14"]]
a = rd("sect15a_hh_w5.dta")[["household_id", "s15q01", "s15q11", "s15q18_1"]]
b = rd("sect15b_hh_w5.dta")[["household_id", "s15q02b", "s15q04", "s15q10a"]]
s = rd("sect5a_hh_w5.dta")
out = []


def wshare(df, col, w="pw_w5", by=None):
    g = df.groupby(by, observed=True) if by else [("all", df)]
    rows = {}
    for k, d in g:
        t = d.groupby(col, observed=True)[w].sum()
        rows[k] = (t / t.sum()).round(3)
    return pd.DataFrame(rows).fillna(0)


hh = cons.merge(a, on="household_id", how="inner")
hh["borrowed"] = hh.s15q01.astype(str).str.contains("Yes", case=False)
hh["refused"] = hh.s15q11.astype(str).str.contains("Yes", case=False)


def w(d, c):
    return (d[c] * d.pw_w5).sum() / d.pw_w5.sum()


t = pd.DataFrame(
    {
        q: {
            "borrowed >=150 birr": w(d, "borrowed"),
            "turned down": w(d, "refused"),
        }
        for q, d in hh.groupby("cons_quint", observed=True)
    }
)
t["all"] = [w(hh, "borrowed"), w(hh, "refused")]
out.append(
    (
        "Households that borrowed in the past 12 months, "
        "by consumption quintile",
        t.round(3),
    )
)
out.append(
    (
        "Why households that did not try to borrow did not (main reason)",
        wshare(hh[hh.s15q18_1.notna()], "s15q18_1").sort_values(
            "all", ascending=False
        ),
    )
)
L = b.merge(cons, on="household_id", how="left")
out.append(
    (
        "Loans by main purpose (share of loans, household-weighted)",
        wshare(L, "s15q04").sort_values("all", ascending=False),
    )
)
out.append(
    (
        "Loans by main purpose and quintile",
        wshare(L, "s15q04", by="cons_quint"),
    )
)
L["pw_amt"] = L.pw_w5 * pd.to_numeric(L.s15q10a, errors="coerce").fillna(0)
out.append(
    (
        "Loans by main purpose (share of principal, where reported)",
        wshare(L[L.pw_amt > 0], "s15q04", w="pw_amt").sort_values(
            "all", ascending=False
        ),
    )
)
out.append(
    (
        "Loans by lender (share of loans)",
        wshare(L, "s15q02b").sort_values("all", ascending=False),
    )
)
# saving: individuals (adults answering)
S = s[s.s5aq09.notna()].merge(
    cons[["household_id", "cons_quint"]], on="household_id", how="left"
)
S["saved"] = S.s5aq09.astype(str).str.contains("Yes", case=False)
fcols = [c for c in S if c.startswith("s5aq11_") and c[-1].isdigit()]
icols = [c for c in S if c.startswith("s5aq13_") and c[-1].isdigit()]
S["formal"] = S[fcols].apply(
    lambda r: r.astype(str).str.contains("Yes", case=False).any(), axis=1
)
S["informal"] = S[icols].apply(
    lambda r: r.astype(str).str.contains("Yes", case=False).any(), axis=1
)
t = pd.DataFrame(
    {
        q: {k: w(d, k) for k in ["saved", "formal", "informal"]}
        for q, d in S.groupby("cons_quint", observed=True)
    }
)
t["all"] = [w(S, k) for k in ["saved", "formal", "informal"]]
out.append(
    (
        "Adults who saved in the past 12 months, any / formal (banks, "
        "MFI, SACCO) / informal (home, family, association, equb)",
        t.round(3),
    )
)
out.append(
    (
        "Main reason for saving (savers)",
        wshare(S[S.saved], "s5aq10").sort_values("all", ascending=False),
    )
)
out.append(
    (
        "Main reason for saving by quintile (savers)",
        wshare(S[S.saved], "s5aq10", by="cons_quint"),
    )
)
txt = []
for title, df in out:
    txt += [f"## {title}", "", df.to_string(), ""]
txt = "\n".join(txt)
print(txt)
(OUT / "credit_saving.txt").write_text(txt)
print("n households", len(hh), "loans", len(L), "adults", len(S))
