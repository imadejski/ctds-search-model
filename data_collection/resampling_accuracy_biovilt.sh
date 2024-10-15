#!/bin/bash

# Define the base directory
BASE_DIR="/opt/gpudata/imadejski/search-model/ctds-search-model/data/param_search_v3_biovilt"

# Loop through each subdirectory in the base directory
for SUB_DIR in "$BASE_DIR"/model_run_*; do
  # Define the input and output file paths
  COSINE_PATH="$SUB_DIR/cosine_similarity.csv"
  OUTPUT_RESULTS_PATH="$SUB_DIR/top_n_accuracy_results.csv"
  OUTPUT_TOP_K_RESULTS_PATH="$SUB_DIR/top_k_accuracy_results.csv"
  
  # Run the resampled accuracy script with the required arguments
  python /opt/gpudata/imadejski/search-model/ctds-search-model/data_collection/general_mimic_accuracy_resampling.py \
    -c "$COSINE_PATH" \
    -o "$OUTPUT_RESULTS_PATH" \
    -t "$OUTPUT_TOP_K_RESULTS_PATH" \
    -s "validate" \
    -n 1000 \
    -r
done