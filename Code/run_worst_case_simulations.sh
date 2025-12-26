#!/bin/bash

# Shell script to run worst_case_sims.py with different methods and parameters
# Usage: ./run_worst_case_simulations.sh

echo "========================================"
echo "WORST CASE SIMULATIONS"
echo "========================================"

echo ""
echo "Step 1: Generating simulation data..."
python worst_case_sims.py --verbose --store-data

echo ""
echo "Step 2: Running Rashomon method..."
python worst_case_sims.py --methods rashomon --verbose

echo ""
echo "Step 3: Running Lasso method..."
python worst_case_sims.py --methods lasso --verbose

echo ""
echo "Step 4: Running TVA method..."
python worst_case_sims.py --methods tva --verbose

echo ""
echo "Step 5: Running Bayesian Lasso method..."
python worst_case_sims.py --methods blasso --verbose

echo ""
echo "Step 6: Running Bootstrap method..."
python worst_case_sims.py --methods bootstrap --verbose

echo ""
echo "Step 7: Running Spike-Slab Lasso method..."
python worst_case_sims.py --methods ssl --verbose

echo ""
echo "Step 8: Running PPMx method (R)..."
Rscript R/worst_case_ppmx_sims.R

echo ""
echo "========================================"
echo "ALL SIMULATIONS COMPLETE"
echo "========================================"