#!/bin/bash

# Shell script to run simulations.py with different methods and parameters
# Usage: ./run_reff_simulations.sh

# Parameters
PARAMS_FILE="reff_params"
OUTPUT_PREFIX="reff"
SAMPLE_SIZE=30
ITERS=100

echo "========================================"
echo "RASHOMON EFFECT SIMULATIONS"
echo "========================================"

echo "================================"
echo "Starting simulations"
echo "Parameters file: $PARAMS_FILE"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Sample size: $SAMPLE_SIZE"
echo "Iterations: $ITERS"
echo "================================"
echo ""


echo ""
echo "Step 1: Generating simulation data..."
python reff_simulations.py --params $PARAMS_FILE --verbose --store-data

echo ""
echo "Step 2: Running Rashomon method..."
python reff_simulations.py --params $PARAMS_FILE --methods r --sample_size $SAMPLE_SIZE --iters $ITERS --output_prefix $OUTPUT_PREFIX --verbose

echo ""
echo "Step 3: Running Lasso method..."
python reff_simulations.py --params $PARAMS_FILE --methods lasso --sample_size $SAMPLE_SIZE --iters $ITERS --output_prefix $OUTPUT_PREFIX --verbose

echo ""
echo "Step 4: Running Bayesian Lasso method..."
python reff_simulations.py --params $PARAMS_FILE --methods blasso --sample_size $SAMPLE_SIZE --iters $ITERS --output_prefix $OUTPUT_PREFIX --verbose

echo ""
echo "Step 5: Running Bootstrap method..."
python reff_simulations.py --params $PARAMS_FILE --methods bootstrap --sample_size $SAMPLE_SIZE --iters $ITERS --output_prefix $OUTPUT_PREFIX --verbose

echo ""
echo "Step 6: Running Spike-Slab Lasso method..."
python reff_simulations.py --params $PARAMS_FILE --methods ssl --sample_size $SAMPLE_SIZE --iters $ITERS --output_prefix $OUTPUT_PREFIX --verbose

echo ""
echo "Step 7: Running PPMx method (R)..."
Rscript R/reff_ppmx_sims.R

echo ""
echo "========================================"
echo "ALL SIMULATIONS COMPLETE"
echo "========================================"