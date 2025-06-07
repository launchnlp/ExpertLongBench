# ExpertLongBench

Code for ExpertLongBench.

## Requirements

Please install requirements using `requirements.txt`.

In addition, for running proprietary models, we require the following environment variables and settings. We use batch prediction APIs provided by OpenAI (for GPT) and Google Vertex AI (for Gemini & Claude).
- `OPENAI_API_KEY` for running GPT-4o and GPT-4o-mini via OpenAI.
- Set up Google Cloud authorization with `gcloud auth application-default login` and make sure you have enabled Vertex AI as well as the Claude model.
- Set up Google Cloud Storage Bucket. Set `GCS_BUCKET_NAME` to the bucket name for storing the batch prediction files.
- `ANTHROPIC_API_KEY` for counting number of tokens in input to do truncation.

## Getting Task Outputs from Model

### Prepare Data

Our data contains private samples. Please contact the authors for obtaining data. The data should be placed inside `exp/data`. Create the folder if it does not exist.

```
./shell_scripts/create_task_output_requests.sh
```

### Get Model Outputs

#### Proprietary Models

Submit batch prediction job.

```
./shell_scripts/submit_task_output_proprietary.sh
```

Download batch prediction job.

```
./shell_scripts/download_task_output_proprietary.sh
```

#### Open-weight Models

```
./shell_scripts/inference_task_output_openweight.sh
```

## Checklist Mapping

The shell scripts that automatically extracts the checklist from the model outputs would be uploaded later. For now, we provide the code for each step of the checklist mapping from a particular model's output.

### Generating request for checklist mapping
The first step is to generate a request file that can be used by the model inference code to generate the checklist mapping. This request file is essentially a JSONL file where each line represents the request to map a particular checklist item from a particular sample.
```
python extraction/checklist_mapper_request.py \
    --model_output <Path to the model output for a particular model and a task generated in the previous step> \
    --task_name <Task ID of the model output (e.g., "T1LegalMDS")> \
    --output_file <Path to the output file where the request will be saved>
```

### Generating output for the checklist mapping request
The second step is to generate the output for the request file generated in the previous step. This output file is also a JSONL file where each line represents the model's response to a particular request.
```
python model_running_scripts/run_model.py \
    --model_config configs/models/Qwen2.5-72B-Instruct.yaml \
    --data_file <Path to the request file generated in the previous step> \
    --output_file <Path to the output file where the model's response will be saved> \
    --vllm_offline
```

### Post-processing the checklist mapping output
The final step is to post-process the output generated in the previous step. This step will convert the model's response into a structured format that can be used for evaluation. This format is a dictionary where each key is a sample ID which maps to a dictionary. This dictionary contains the checklist items as keys and their corresponding values as values.
```
python extraction/process_checklist_output.py \
    --checklist_output_file <Path to the output file generated in the previous step> \
    --checklist_request_file <Path to the request file generated in the first step> \
    --parsed_output <Path to the output file where the processed checklist will be saved>
```