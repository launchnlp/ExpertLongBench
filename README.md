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