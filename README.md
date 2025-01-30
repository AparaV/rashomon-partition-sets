# Rashomon Partition Sets 

This code is supplementary to Venkateswaran, A., Sankar, A., Chandrasekhar, A. G., & McCormick, T. H. (2024). Robustly estimating heterogeneity in factorial data using Rashomon Partitions. _arXiv preprint arXiv:2404.02141_. URL: [https://arxiv.org/abs/2404.02141](https://arxiv.org/abs/2404.02141)

## Requirements

- Python >= 3.11

## Setup

1. Clone the repo
```
$ git clone https://github.com/AparaV/rashomon-partition-sets.git
$ cd rashomon-partition-sets/Code
```

2. Setup the virtual environment
```
$ python -m venv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

## Developer instructions

To update `requirements.txt`, run
```
$ pip install pipreqs
$ pipreqs . --force
```


## Data

All of the data require the researcher to agree to terms and conditions. So the data is not publicly released as a part of this repository. Below are instructions for where one can find and obtain the same data.

### Charitable donations

You can find the data at Karlan, Dean; List, John A., 2014, "Does Price Matter in Charitable Giving? Evidence from a Large-Scale Natural Field Experiment", https://doi.org/10.7910/DVN/27853, Harvard Dataverse, V4, UNF:5:C63Hp69Cn88CkFCoOA0N/w== [fileUNF]. The URL is https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/27853&version=4.2 and the key file is `AER_merged.dta`.

In `Code/real_data_charitable_donations.ipynb`, this file has been renamed to `Does_Price_Matter_AER_merged.dta` for organization purposes.

### Microfinance

You can find this dataset at https://www.openicpsr.org/openicpsr/project/113599/version/V1/view. The key file is `2013-0533_data_endlines1and2.dta`. In `Code/process_microfinance_data.py`, this file has been renamed to `banerjee_mircale_endlines_raw.dta` for organization purposes.  

### NHANES

The NHANES data is obtained from _National Health and Nutrition Examination Survey (NHANES), 1999-2000 (ICPSR 25501) and 2001-2002 (ICPSR 25502)_. You can download the files containing the covariates from https://www.icpsr.umich.edu/web/NACDA/studies/25501/publications and https://www.icpsr.umich.edu/web/NACDA/studies/25502/versions/V5 respectively. Please download the STATA version. The key files are `ICPSR_25501/DS0228/25501-0228-Data.dta` and `ICPSR_25502/DS0229/25502-0229-Data.dta` respectively.

The telomere data is obtained from https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&Cycle=1999-2000 (search for `TELO_A_Data` -- it is an `XPT` file). See https://wwwn.cdc.gov/Nchs/Nhanes/1999-2000/TELO_A.htm for documentation. The other half from 2001-2002 can be obtained similarly from https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&Cycle=2001-2002 (search for `TELO_B_Data` -- it is an `XPT` file). See https://wwwn.cdc.gov/Nchs/Nhanes/2001-2002/TELO_B.htm for documentation.

See `Code/process_nhanes_data.py` for processing the data before consuming it.


## Contact

For any questions, please contact Apara Venkat (apara.vnkat@gmail.com) or Tyler McCormick (tylermc@uw.edu). Feel free to also open an issue on GitHub.