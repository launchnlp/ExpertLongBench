# GPT
for MODEL in gpt-4o-2024-11-20 gpt-4o-mini-2024-07-18
do
    for DIR in exp/all_requests/task_outputs
    do
        for FILE in "$DIR"/*.jsonl
        do
            TASKNAME=$(basename "$FILE" .jsonl)
            BASE_OUTPUT_DIR=${DIR/all_requests/all_results}
            OUTPUT_DIR=${BASE_OUTPUT_DIR}/${MODEL}
            BASE_COST_DIR=${DIR/all_requests/all_costs}
            COST_DIR=${BASE_COST_DIR}/${MODEL}
            BASE_LOG_DIR=${DIR/all_requests/request_logs}
            LOG_DIR=${BASE_LOG_DIR}/${MODEL}
            mkdir -p "$OUTPUT_DIR"; mkdir -p "$COST_DIR"
            python model_running_scripts/download_openai.py --log_file "${LOG_DIR}/${TASKNAME}.json" --output_file "${OUTPUT_DIR}/${TASKNAME}.jsonl" --cost_file "${COST_DIR}/${TASKNAME}.jsonl"
        done
    done
done

# Gemini
for MODEL in gemini-2.0-flash-001
do
    for DIR in exp/all_requests/task_outputs
    do
        for FILE in "$DIR"/*.jsonl
        do
            TASKNAME=$(basename "$FILE" .jsonl)
            BASE_OUTPUT_DIR=${DIR/all_requests/all_results}
            OUTPUT_DIR=${BASE_OUTPUT_DIR}/${MODEL}
            BASE_COST_DIR=${DIR/all_requests/all_costs}
            COST_DIR=${BASE_COST_DIR}/${MODEL}
            BASE_LOG_DIR=${DIR/all_requests/request_logs}
            LOG_DIR=${BASE_LOG_DIR}/${MODEL}
            mkdir -p "$OUTPUT_DIR"; mkdir -p "$COST_DIR"
            python model_running_scripts/download_gcp.py --log_file "${LOG_DIR}/${TASKNAME}.json" --output_file "${OUTPUT_DIR}/${TASKNAME}.jsonl" --cost_file "${COST_DIR}/${TASKNAME}.jsonl"
        done
    done
done

# Claude
for MODEL in claude-3.7-sonnet-20250219 claude-3.5-haiku-20241022
do
    for DIR in exp/all_requests/task_outputs
    do
        for FILE in "$DIR"/*.jsonl
        do
            TASKNAME=$(basename "$FILE" .jsonl)
            BASE_OUTPUT_DIR=${DIR/all_requests/all_results}
            OUTPUT_DIR=${BASE_OUTPUT_DIR}/${MODEL}
            BASE_COST_DIR=${DIR/all_requests/all_costs}
            COST_DIR=${BASE_COST_DIR}/${MODEL}
            BASE_LOG_DIR=${DIR/all_requests/request_logs}
            LOG_DIR=${BASE_LOG_DIR}/${MODEL}
            mkdir -p "$OUTPUT_DIR"; mkdir -p "$COST_DIR"
            python model_running_scripts/download_claude.py --log_file "${LOG_DIR}/${TASKNAME}.json" --output_file "${OUTPUT_DIR}/${TASKNAME}.jsonl" --cost_file "${COST_DIR}/${TASKNAME}.jsonl"
        done
    done
done