#!/bin/bash

outcome_col=19
q=0.0031785
lambda=1.5e-6


python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda --edu
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda --gen
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda --edu --gen
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda --trt
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda --trt --edu
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda --trt --gen
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $lambda --trt --edu --gen

# python run_microfinance_te.py --outcome_col $outcome_col