import numpy as np
import pandas as pd


# Filenames
occ_df_1999_fname = "../../datasets/ICPSR_25501/DS0228/25501-0228-Data.dta"
occ_df_2001_fname = "../../datasets/ICPSR_25502/DS0229/25502-0229-Data.dta"
telo_a_fname = "../../datasets/telomere/TELO_A.XPT"
telo_b_fname = "../../datasets/telomere/TELO_B.XPT"
output_fname = "../Data/NHANES_telomere_std.csv"

# Read occuptation databases
occ_df_1999 = pd.read_stata(occ_df_1999_fname)
occ_df_2001 = pd.read_stata(occ_df_2001_fname)

# Remove unnecessary columns
columns_to_keep_1999 = ["SEQN", "OCQ180", "RIAGENDR", "RIDAGEYR", "RIDRETH1",
                        "DMDEDUC3", "DMDEDUC2", "DMDMARTL", "INDHHINC"]
columns_to_keep_2001 = ["SEQN", "OCD180", "RIAGENDR", "RIDAGEYR", "RIDRETH1",
                        "DMDEDUC3", "DMDEDUC2", "DMDMARTL", "INDHHINC"]
col_map = {
    "SEQN": "SEQN",
    "OCQ180": "HoursWorked",
    "OCD180": "HoursWorked",
    "RIAGENDR": "Gender",
    "RIDAGEYR": "Age",
    "RIDRETH1": "Race",
    "DMDEDUC3": "Education",
    "DMDEDUC2": "Education2",
    "DMDMARTL": "MaritalStatus",
    "INDHHINC": "HouseholdIncome"
}

occ_df_1999 = occ_df_1999[columns_to_keep_1999]
occ_df_1999 = occ_df_1999.rename(columns=col_map)
occ_df_2001 = occ_df_2001[columns_to_keep_2001]
occ_df_2001 = occ_df_2001.rename(columns=col_map)

# Concatenate dataframes
occ_df = pd.concat([occ_df_1999, occ_df_2001], axis=0)

# Merge education columns
occ_df["Education"] = occ_df["Education"].fillna(occ_df["Education2"])
occ_df = occ_df.drop(columns=["Education2"])

# Remap values

occ_df["Race"] = occ_df["Race"].map({
    "Non-Hispanic White": "White",
    "Non-Hispanic Black": "Black",
    "Mexican American": "Hispanic",
    "Other Hispanic": "Hispanic",
    "Other Race - Including Multi-Racial": "Other"
    }, na_action="ignore")

occ_df["Education"] = occ_df["Education"].map({
    "College Graduate or above": "College",
    "More than high school": "GED",
    "9-11th Grade (Includes 12th grade with no diploma)": "< GED",
    "High School Grad/GED or Equivalent": "GED",
    "5th Grade": "< GED",
    "8th Grade": "< GED",
    "Less Than 9th Grade": "< GED",
    "7th Grade": "< GED",
    "11th Grade": "< GED",
    "Some College or AA degree": "College",
    "High School Graduate": "GED",
    "9th Grade": "< GED",
    "12th Grade, No Diploma": "< GED",
    "6th Grade": "< GED",
    "10th Grade": "< GED",
    "4th Grade": "< GED",
    "GED or Equivalent": "GED",
    "Less Than 5th Grade": "< GED",
    np.nan: "< GED",
    "Don't Know": "< GED",
    "Don't know": "< GED",
    "Refused": "< GED"
    })

occ_df["MaritalStatus"] = occ_df["MaritalStatus"].map({
    "Married": "Married",
    "Never married": "Single",
    "Separated": "Single",
    "Divorced": "Divorced/Widowed",
    "Widowed": "Divorced/Widowed",
    "Living with partner": "Single",
    "Refused": "Single",
    "Don't know": "Single",
    np.nan: "Single"
    })

age_map = {}
for val in pd.unique(occ_df["Age"]):
    if val == ">= 85 years of age":
        val = 85
    if val <= 18:
        age_map[val] = "<=18"
    elif val <= 30:
        age_map[val] = "19-30"
    elif val <= 50:
        age_map[val] = "31-50"
    elif val <= 70:
        age_map[val] = "51-70"
    else:
        age_map[val] = ">=70"
# age_map[np.nan] = "<=18"
occ_df["Age"] = occ_df["Age"].map(age_map)


hrs_worked_map = {}
for val in pd.unique(occ_df["HoursWorked"]):
    if isinstance(val, str):
        val = 0
    # if np.isnan(val):
    #     val = 0
    if val <= 20:
        hrs_worked_map[val] = "<=20"
    elif val <= 40:
        hrs_worked_map[val] = "21-40"
    else:
        hrs_worked_map[val] = ">=41"
occ_df["HoursWorked"] = occ_df["HoursWorked"].map(hrs_worked_map)

income_map = {
    '$75,000 and Over': ">=75k",
    '$65,000 to $74,999': "45k-75k",
    '$55,000 to $64,999': "45k-75k",
    '$45,000 to $54,999': "45k-75k",
    '$35,000 to $44,999': "20k-45k",
    '$25,000 to $34,999': "20k-45k",
    '$20,000 to $24,999': "20k-45k",
    'Over $20,000': "20k-45k",
    'Under $20,000': "<20k",
    '$15,000 to $19,999': "<20k",
    '$10,000 to $14,999': "<20k",
    '$ 5,000 to $ 9,999': "<20k",
    '$ 0 to $ 4,999': "<20k",
    "Don't know": "<20k",
    'Refused': "<20k",
    np.nan: "<20k"
}
occ_df["HouseholdIncome"] = occ_df["HouseholdIncome"].map(income_map)

# Read telomere data
telo_a_df = pd.read_sas(telo_a_fname)
telo_b_df = pd.read_sas(telo_b_fname)

telo_df = pd.concat([telo_a_df, telo_b_df], axis=0)

telo_df = telo_df.dropna()
# telo_df = telo_df.drop(["TELOSTD"], axis=1)
telo_df = telo_df.rename(columns={"TELOMEAN": "Telomean", "TELOSTD": "Telostd"})


# Join dataframes
df = telo_df.join(occ_df, on="SEQN", how="left", lsuffix="_telo", rsuffix="_occ")
df = df.drop(["SEQN", "SEQN_telo", "SEQN_occ"], axis=1)
df = df.dropna()

print(f"There are {len(df)} rows.")

df.to_csv(output_fname, index=None)
