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

Our data contains public and private samples. We recommend starting with the public set for initial testing and development. You're welcome to submit your model for evaluation on the private set — just make sure to include your results on the public set. Refer to the [ExpertLongBench page](https://huggingface.co/spaces/launch/ExpertLongBench) for model submission details.

The data should be placed inside `exp/data`. Create the folder if it does not exist.

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
    --task_name <Task ID of the model output (e.g., "T01LegalMDS")> \
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

## Checklist Evaluation
In this section, we explain the steps required to evaluate the checklist mapping output generated in the previous section.

### Generating the evaluation request
In this step, we generate a request file that can be used by the model inference code to evaluate the checklist mapping output. This request file is essentially a JSONL file where each line represents the request to evaluate a particular checklist item from a particular sample. In each line, we compare against the corresponding ground truth checklist item and determine whether the model output is semantically contained in the ground truth item / ground truth item is semantically contained in the model output.
```
python evaluation/checklist_comparison_request.py \
    --reference_file <Path to the ground truth processed checklist file similar to the one generated in the last step of the previous section> \
    --inference_file <Path to the processed checklist file generated in the last step of the previous section> \
    --output_file <Path to the output file where the request will be saved>
```

### Generating the evaluation output
The evaluation output is essentially a JSONL file where each line represents the model's response to a particular request. Each line contains the model's response to whether the model output is semantically contained in the ground truth item / ground truth item is semantically contained in the model output. There are two sub steps for this step. In first sub step, we generate openai requests and then in the second sub step, we download the model's response.

```
python model_running_scripts/prepare_and_submit_openai.py \
    --model_config configs/models/gpt-4o-2024-11-20.yaml \
    --data_file <Path to the request file generated in the previous step> \
    --log_file <Path to the log file which will be used for second sub-step> \
    --model gpt-4o-2024-11-20
```

```
python model_running_scripts/download_openai.py \
    --log_file <Path to the log file generated in the previous sub step> \
    --output_file <Path to the output file where the model's response will be saved> \
    --cost_file <Path to the cost file where the cost of the model's response will be saved>
```

### Mapping the evaluation output to a structured format
This step does inplace post-processing of the checklist mapped output generated in the previous subsection by incorporating the evaluation output generated in the last step.
```
python checklist_comparison_mapper.py \
    --input_path <Path to the processed checklist file generated in the last step of the previous section> \
    --request_path <Path to the request file generated in the first step of this section> \
    --evaluation_path <Path to the output file generated in the last step of this section>
```
`--variable_checklist` is an optional argument that can be used to specify whether the checklist items are variable and dependent on the data point. If set to `True`, the checklist items are considered variable and the evaluation will be done accordingly.

### Evaluating the checklist mapping output
The final step is to evaluate the checklist mapped output.
```
python evaluation/checklist_comparison_performance.py \
    --input_path <Path to the processed checklist file> \
    --output_path <Path to the output file where the evaluation results will be saved>
```
