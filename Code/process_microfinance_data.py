import pandas as pd


raw_baseline_fname = "../Data/banerjee_miracle_baseline_raw.dta"
raw_endlines_fname = "../Data/banerjee_mircale_endlines_raw.dta"
output_fname = "../Data/banerjee_miracle.csv"

raw_baseline = pd.read_stata(raw_baseline_fname)
raw_endlines = pd.read_stata(raw_endlines_fname)

baseline_cols = [
    "hhid_baseline",
    "male_head",
    "anyloan_amt",
    "female_biz_pct",
    "bizexpense",
    "total_exp_mo",
    "home_durable_index"
]

baseline_df = raw_baseline[baseline_cols]
baseline_df = baseline_df.rename(columns={"hhid_baseline": "hhid"})


baseline_df["bizexpense"] = pd.qcut(
    baseline_df["bizexpense"],
    q=4,
    duplicates="drop",
    labels=[0, 1, 2, 3]
    )
baseline_df["bizexpense"] = baseline_df["bizexpense"].fillna(0)

baseline_df["home_durable_index"] = pd.qcut(
    baseline_df["home_durable_index"],
    q=4,
    duplicates="drop",
    labels=[0, 1, 2, 3]
    )
baseline_df["home_durable_index"] = baseline_df["home_durable_index"].fillna(0)

baseline_df["anyloan_amt"] = pd.qcut(
    baseline_df["anyloan_amt"],
    q=5,
    duplicates="drop",
    labels=[0, 1, 2, 3]
    )
baseline_df["anyloan_amt"] = baseline_df["anyloan_amt"].fillna(0)

baseline_df["total_exp_mo"] = pd.qcut(
    baseline_df["total_exp_mo"],
    q=4,
    duplicates="drop",
    labels=[0, 1, 2, 3]
    )
baseline_df["total_exp_mo"] = baseline_df["total_exp_mo"].fillna(0)

baseline_df[baseline_df["male_head"] == 1] = 2
baseline_df[baseline_df["male_head"] == 0] = 1
baseline_df["male_head"] = baseline_df["male_head"].fillna(1)

baseline_df[baseline_df["female_biz_pct"] > 0] = 2
baseline_df[baseline_df["female_biz_pct"] == 0] = 1
baseline_df["female_biz_pct"] = baseline_df["female_biz_pct"].fillna(1)


endline_cols = raw_endlines.columns
endline2_cols = [x for x in endline_cols if x[-1] == "2"] + ["hhid", "treatment"]
endline2_df = raw_endlines[endline2_cols].copy()

endline2_df["treatment"] = endline2_df["treatment"].replace({
    "Treatment": 1,
    "Control": 0
})

endline2_cols_to_keep = [
    "hhid",
    "treatment",
    "anyloan_amt_2",
    "informal_amt_2",
    "female_biz_pct_2",   # Number of women owned business
    "hours_week_2",
    "durables_exp_mo_2",
    "temptation_exp_mo_2",
    "total_exp_mo_2",
    "girl515_school_2",
    "girl1620_school_2",
    "bizprofit_2",
    "bizrev_2",
    "bizemployees_2"
]

endline2_clean_df = endline2_df[endline2_cols_to_keep].copy()

endline2_clean_df["girls_school"] = endline2_clean_df.loc[:, ("girl515_school_2", "girl1620_school_2")].mean(axis=1)
endline2_clean_df = endline2_clean_df.drop(["girl515_school_2", "girl1620_school_2"], axis=1)


# Merge dataframes
df = baseline_df.join(endline2_clean_df, on="hhid", how="left", lsuffix="_0", rsuffix="_1")
df = df.drop(["hhid_0", "hhid_1"], axis=1)

print(f"There are {len(df)} rows.")

df.to_csv(output_fname, index=None)
