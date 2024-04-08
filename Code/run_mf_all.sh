#!/bin/bash

outcome_col=25
q=0.00302
lambda=1e-5


python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda --edu
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda --gen
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda --edu --gen
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda --trt
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda --trt --edu
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda --trt --gen
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda --trt --edu --gen

# python run_microfinance_te.py --outcome_col $outcome_col