#!/bin/bash

# Define the base paths and split type
model_base_path="/opt/gpudata/imadejski/image_text_global_local_alignment/param_search_v3_biovilt"
output_base_path="/opt/gpudata/imadejski/search-model/ctds-search-model/data/param_search_v3_biovilt"
split_type="validate"

# Define the model configurations
model_configs=("model_run_4_False_igl_tgl" "model_run_5_False_igl_tg" "model_run_6_False_ig_tgl" "model_run_7_False_ig_tg")

# Loop through each model configuration and run the Python scripts
for model in "${model_configs[@]}"; do
    model_checkpoint_path="${model_base_path}/${model}"
    output_path="${output_base_path}/${model}"

    echo "Running scripts for:"
    echo "  Model checkpoint path: $model_checkpoint_path"
    echo "  Output path: $output_path"
    echo "  Split type: $split_type"

    # Create the output directory if it doesn't exist
    mkdir -p "$output_path"

    # Check if embedding_library.csv exists
    if [ ! -f "$output_path/embedding_library.csv" ]; then
        # Run general_mimic_embedding_library.py
        python data_collection/general_mimic_embedding_library.py "$model_checkpoint_path" "$output_path/embedding_library.csv" "$split_type"
    else
        echo "  Skipping general_mimic_embedding_library.py: embedding_library.csv already exists."
    fi

    # Check if cosine_similarity.csv exists
    if [ ! -f "$output_path/cosine_similarity.csv" ]; then
        # Run general_mimic_cosine_similarity.py
        python data_collection/general_mimic_cosine_similarity.py "$model_checkpoint_path" "$output_path/embedding_library.csv" "$output_path/cosine_similarity.csv" "$split_type"
    else
        echo "  Skipping general_mimic_cosine_similarity.py: cosine_similarity.csv already exists."
    fi

    # Run general_mimic_accuracy.py
    python data_collection/general_mimic_accuracy.py "$output_path/cosine_similarity.csv" "$output_path/model_top_n_accuracy_results.csv" "$output_path/model_top_k_accuracy_results.csv" "$split_type"
done

echo "All scripts have been run successfully."