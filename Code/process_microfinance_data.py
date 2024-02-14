import pandas as pd


raw_endlines_fname = "../Data/banerjee_mircale_endlines_raw.dta"
output_fname = "../Data/banerjee_miracle.csv"

raw_endlines = pd.read_stata(raw_endlines_fname)

#
# Subset dataframe
#
regional_cols = [
    "hhid",
    "treatment",
    "areaid",
    "old_biz",
    "area_pop_base",
    "area_debt_total_base",
    "area_business_total_base",
    "area_exp_pc_mean_base",
    "area_literate_head_base",
    "area_literate_base"
]

endline2_cols = [
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

cols_to_keep = endline2_cols + regional_cols

endline2_df = raw_endlines[cols_to_keep].copy()

#
# Clean up the columns
#
endline2_df["girls_school"] = endline2_df.loc[:, ("girl515_school_2", "girl1620_school_2")].mean(axis=1)
endline2_df = endline2_df.drop(["girl515_school_2", "girl1620_school_2"], axis=1)

endline2_df["treatment"] = endline2_df["treatment"].replace({
    "Treatment": 1,
    "Control": 0
})

# Discretize continuous covariates
endline2_df["old_biz"] = endline2_df["old_biz"].fillna(0)
endline2_df["old_biz"] = endline2_df["old_biz"].replace({
    0: 0,
    1: 1,
    2: 1,
    3: 2,
    4: 2,
    5: 2,
    6: 3,
    7: 3,
    8: 3
})

qcut_cols = [
    "area_pop_base",
    "area_debt_total_base",
    "area_business_total_base",
    "area_exp_pc_mean_base",
    "area_literate_head_base",
    "area_literate_base"
]

for qcut_col in qcut_cols:
    endline2_df[qcut_col] = pd.qcut(
        endline2_df[qcut_col],
        q=4,
        duplicates="drop",
        labels=[0, 1, 2, 3]
    )

#
# Reorder columns
#
reordered_cols = [
    "hhid",
    "areaid",
    "treatment",
    "old_biz",
    "area_pop_base",
    "area_debt_total_base",
    "area_business_total_base",
    "area_exp_pc_mean_base",
    "area_literate_head_base",
    "area_literate_base",
    "anyloan_amt_2",
    "informal_amt_2",
    "female_biz_pct_2",
    "hours_week_2",
    "durables_exp_mo_2",
    "temptation_exp_mo_2",
    "total_exp_mo_2",
    "bizprofit_2",
    "bizrev_2",
    "bizemployees_2"
]

endline2_df = endline2_df[reordered_cols]

#
# Save to CSV
#
print(f"There are {len(endline2_df)} rows.")

endline2_df.to_csv("../Data/banerjee_miracle.csv", index=None)
