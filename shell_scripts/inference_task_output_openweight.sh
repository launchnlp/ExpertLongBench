for MODEL in Qwen2.5-7B-Instruct Mistral-Nemo-Instruct-2407 Llama-3.1-8B-Instruct Qwen2.5-72B-Instruct Mistral-Large-Instruct-2411 Llama-3.3-70B-Instruct
do
  DATA_FILES=()
  OUTPUT_FILES=()

  for DIR in exp/all_requests/task_outputs
  do
    for FILE in "$DIR"/*.jsonl
    do
      TASKNAME=$(basename "$FILE" .jsonl)
      BASE_OUTPUT_DIR=${DIR/all_requests/all_results}
      OUTPUT_DIR=${BASE_OUTPUT_DIR}/${MODEL}
      mkdir -p "$OUTPUT_DIR"

      DATA_FILES+=("$FILE")
      OUTPUT_FILES+=("${OUTPUT_DIR}/${TASKNAME}.jsonl")
    done
  done
  DATA_ARGS=()
  for DF in "${DATA_FILES[@]}"; do
    DATA_ARGS+=("--data_file" "$DF")
  done

  OUTPUT_ARGS=()
  for OF in "${OUTPUT_FILES[@]}"; do
    OUTPUT_ARGS+=("--output_file" "$OF")
  done

  python model_running_scripts/run_model.py \
    --model_config "configs/models/${MODEL}.yaml" \
    "${DATA_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    --vllm_offline
done