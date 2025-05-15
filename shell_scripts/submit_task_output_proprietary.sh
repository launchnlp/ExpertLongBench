# GPT
for MODEL in gpt-4o-mini-2024-07-18 gpt-4o-2024-11-20
do
    for DIR in exp/all_requests/task_outputs
    do
        for FILE in "$DIR"/*.jsonl
        do
        TASKNAME=$(basename "$FILE" .jsonl)
        BASE_LOG_DIR=${DIR/all_requests/request_logs}
        LOG_DIR=${BASE_LOG_DIR}/${MODEL}
        mkdir -p "$LOG_DIR"
        python model_running_scripts/prepare_and_submit_openai.py --model_config configs/models/${MODEL}.yaml --data_file "${FILE}" --log_file "${LOG_DIR}/${TASKNAME}.json"
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
            BASE_LOG_DIR=${DIR/all_requests/request_logs}            
            LOG_DIR=${BASE_LOG_DIR}/${MODEL}      
            mkdir -p "$LOG_DIR"                                            
            python model_running_scripts/prepare_and_submit_gcp.py --model_config configs/models/${MODEL}.yaml --data_file "${FILE}" --log_file "${LOG_DIR}/${TASKNAME}.json" --gcs_bucket_name ${GCS_BUCKET_NAME} --gcs_data_path model_running_requests/${MODEL}/${FILE} --gcs_output_path model_running_requests/${MODEL}/${FILE}                  
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
            BASE_LOG_DIR=${DIR/all_requests/request_logs}            
            LOG_DIR=${BASE_LOG_DIR}/${MODEL}      
            mkdir -p "$LOG_DIR"                                            
            python model_running_scripts/prepare_and_submit_claude.py --model_config configs/models/${MODEL}.yaml --data_file "${FILE}" --log_file "${LOG_DIR}/${TASKNAME}.json" --gcs_bucket_name ${GCS_BUCKET_NAME} --gcs_data_path model_running_requests/${MODEL}/${FILE} --gcs_output_path model_running_requests/${MODEL}/${FILE}               
        done                                        
    done
done 