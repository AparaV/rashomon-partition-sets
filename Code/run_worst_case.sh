#!/bin/bash

# Shell script to run worst_case_sims.py with different methods and parameters
# Usage: ./run_worst_case.sh [--test]

# Check if test flag is provided
TEST_FLAG=""
if [ "$1" == "--test" ]; then
    TEST_FLAG="--test"
    echo "Running in TEST mode"
fi

# Methods to run (can be modified to run specific methods)
# Options: rashomon, lasso, tva, blasso, bootstrap, ppmx
METHODS=("rashomon" "lasso" "tva" "blasso" "bootstrap" "ppmx")

# Sample sizes (overridden in test mode to [10, 50])
SAMPLE_SIZES=(10 20 50 100 500 1000)

# Number of iterations (overridden in test mode to 5)
ITERS=100

# Output suffix (optional, can be used to distinguish runs)
OUTPUT_SUFFIX=""

echo "================================"
echo "Starting worst case simulations"
echo "Methods: ${METHODS[@]}"
if [ -z "$TEST_FLAG" ]; then
    echo "Sample sizes: ${SAMPLE_SIZES[@]}"
    echo "Iterations: $ITERS"
else
    echo "Sample sizes: [10, 50] (test mode)"
    echo "Iterations: 5 (test mode)"
fi
echo "================================"
echo ""

# Build methods argument string
METHODS_ARG=""
for method in "${METHODS[@]}"; do
    METHODS_ARG="$METHODS_ARG $method"
done

# Run simulations
if [ -z "$TEST_FLAG" ]; then
    # Normal mode - specify sample sizes and iterations
    python worst_case_sims.py \
        --methods $METHODS_ARG \
        --samples ${SAMPLE_SIZES[@]} \
        --iters "$ITERS" \
        --output_suffix "$OUTPUT_SUFFIX"
else
    # Test mode - use defaults (2 sample sizes, 5 iterations)
    python worst_case_sims.py \
        --methods $METHODS_ARG \
        --test \
        --output_suffix "$OUTPUT_SUFFIX"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "✓ Worst case simulations completed successfully!"
    echo "================================"
else
    echo ""
    echo "================================"
    echo "✗ Simulations failed with error code $?"
    echo "================================"
    exit 1
fi
