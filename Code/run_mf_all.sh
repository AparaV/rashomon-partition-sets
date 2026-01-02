#!/bin/bash

outcome_col_idx=25

OUTCOME_COLUMNS=(14 15 16 17 18 19 20 21 22 23 24 25)
Q_VALUES=(0.003458 0.0023235 0.148002 0.0064842 0.0033315 0.0031785 0.0030705 0.001347 0.007559 0.0012185 0.1185375 0.00276)
PRUNED_EPSILON=('2e-04' '4e-04' '0' '2.5e-05' '2e-04' '4e-04' '2e-04' '1.5e-03' '1e-04' '1.25e-03' '0' '5e-04')
LAMBDA=1.5e-6

outcome_col=${OUTCOME_COLUMNS[$outcome_col_idx-14]}
q=${Q_VALUES[$outcome_col_idx-14]}
desired_eps=${PRUNED_EPSILON[$outcome_col_idx-14]}

python run_microfinance.py --outcome_col $outcome_col --q $q --reg $LAMBDA
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $LAMBDA --edu
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $LAMBDA --gen
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $LAMBDA --edu --gen
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $LAMBDA --trt
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $LAMBDA --trt --edu
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $LAMBDA --trt --gen
python run_microfinance.py --outcome_col $outcome_col --q $q --reg $LAMBDA --trt --edu --gen

python run_microfinance_pruning.py --outcome_col $outcome_col --reg $LAMBDA --eps $desired_eps

python run_microfinance_te.py --outcome_col $outcome_col