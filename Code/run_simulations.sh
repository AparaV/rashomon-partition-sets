#!/bin/bash

# Shell script to run simulations.py with different methods and parameters
# Usage: ./run_simulations.sh [--test]

# Check if test flag is provided
TEST_FLAG=""
if [ "$1" == "--test" ]; then
    TEST_FLAG="--test"
    echo "Running in TEST mode"
fi

# Parameters
PARAMS_FILE="reff_4"
OUTPUT_PREFIX="4arms"

# Sample sizes to test (can be modified)
SAMPLE_SIZES=(30)

# Number of iterations (can be modified, overridden in test mode)
ITERS=100

# Methods to run
# METHODS=("lasso" "blasso" "bootstrap" "ppmx" "r")
METHODS=("blasso")

echo "================================"
echo "Starting simulations"
echo "Parameters file: $PARAMS_FILE"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Sample sizes: ${SAMPLE_SIZES[@]}"
echo "Iterations: $ITERS"
echo "Methods: ${METHODS[@]}"
echo "================================"
echo ""

# Run simulations for each method and sample size
for method in "${METHODS[@]}"; do
    echo "Running method: $method"
    
    for sample_size in "${SAMPLE_SIZES[@]}"; do
        echo "  Sample size: $sample_size"
        
        python simulations.py \
            --params "$PARAMS_FILE" \
            --sample_size "$sample_size" \
            --iters "$ITERS" \
            --output_prefix "$OUTPUT_PREFIX" \
            --method "$method" \
            $TEST_FLAG
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Completed: $method with n=$sample_size"
        else
            echo "  ✗ Failed: $method with n=$sample_size"
        fi
        echo ""
    done
    
    echo "Completed all sample sizes for: $method"
    echo "---"
    echo ""
done

echo "================================"
echo "All simulations completed!"
echo "================================"
