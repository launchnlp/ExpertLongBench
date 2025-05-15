import json
import argparse
import yaml
import tempfile
import os
from dataclasses import dataclass, fields
from google import genai
from google.cloud import storage
from google.genai.types import CreateBatchJobConfig


@dataclass
class Example:
    id: str
    system_prompt: str
    sample: str
    max_output_tokens: int

    @classmethod
    def from_json(cls, json_data):
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in json_data.items() if k in field_names}
        if "max_output_tokens" not in filtered_data:
            filtered_data["max_output_tokens"] = json_data["max_tokens"]
        return cls(**filtered_data)


def truncate(text, max_tokens, truncation_factor=0.5):
    # 100 tokens ~= 70 words
    approximate_max_tokens = int(max_tokens * truncation_factor)
    return " ".join(text.split(" ")[:approximate_max_tokens])


def prepare_upload_submit_batch_file(
    data_file,
    log_file,
    gcs_bucket_name,
    gcs_data_path,
    gcs_output_path,
    model,
    temperature=0.0,
    top_p=1.0,
    max_tokens=1000000,
    estimate_cost=False,
):
    client = genai.Client(
        vertexai=True,
        location="us-central1",
    )

    gcs_client = storage.Client()
    bucket = gcs_client.bucket(gcs_bucket_name)

    with open(data_file) as f:
        samples = [Example.from_json(json.loads(line)) for line in f]

    if estimate_cost:
        input_tokens = 0
        output_tokens = 0
        for sample in samples:
            max_output_tokens = sample.max_output_tokens
            if "gemini-2.0-flash" in model:
                max_output_tokens = min(max_output_tokens, 8192)
            input = truncate(sample.sample, max_tokens - max_output_tokens - 1024, truncation_factor)
            response = client.models.count_tokens(
                model=model,
                contents=input,
            )
            input_tokens += response.total_tokens
            output_tokens += sample.max_output_tokens
        
        if model == "gemini-2.0-flash-001":
            cost_per_1k_input = 0.15 / 1000
            cost_per_1k_output = 0.6 / 1000
        else:
            raise ValueError(f"Unknown model {model} for cost estimation.")
        total_input_cost = (input_tokens / 1000) * cost_per_1k_input / 2
        total_output_cost = (output_tokens / 1000) * cost_per_1k_output / 2
        total_cost = total_input_cost + total_output_cost
        print(f"Estimated cost for {len(samples)} samples:")
        print(f"  Input tokens: {input_tokens} -> ${total_input_cost:.4f}")
        print(f"  Output tokens: {output_tokens} -> ${total_output_cost:.4f}")
        print(f"  Total cost: ${total_cost:.4f}")
        return
    
    sample_ids = []
    with tempfile.NamedTemporaryFile("w") as temp_file:
        for sample in samples:
            max_output_tokens = sample.max_output_tokens
            if "gemini-2.0-flash" in model:
                max_output_tokens = min(max_output_tokens, 8192)
            truncation_factor = 0.5
            while True:
                input = truncate(sample.sample, max_tokens - max_output_tokens - 1024, truncation_factor)
                response = client.models.count_tokens(
                    model=model,
                    contents=input,
                )
                if response.total_tokens <= max_tokens:
                    break
                truncation_factor -= 0.1
                if truncation_factor < 0.1:
                    raise ValueError(f"Input text is too long after truncation. Please reduce the input size.")

            batch_sample = {
                "request": {
                    "contents": [
                        {"role": "user", "parts": [{"text": input}]}
                    ],
                    "systemInstruction": {"role": "system","parts": [{"text": sample.system_prompt}]},
                    "generationConfig": {
                        "maxOutputTokens": max_output_tokens,
                        "temperature": temperature,
                        "topP": top_p,
                    },
                    "labels": {
                        "custom_id": sample.id,
                    }
                }
            }
            id = sample.id
            sample_ids.append(id)

            temp_file.write(json.dumps(batch_sample) + "\n")
        temp_file.flush()

        blob = bucket.blob(gcs_data_path)
        blob.upload_from_filename(temp_file.name)

    gcs_data_path = f"gs://{gcs_bucket_name}/{gcs_data_path}"

    print("Uploaded data file to GCS:", gcs_data_path)

    gcs_batch_job = client.batches.create(
        model=model,
        src=gcs_data_path,
        config=CreateBatchJobConfig(dest=f"gs://{gcs_bucket_name}/{gcs_output_path}"),
    )

    with open(log_file, "w") as f:
        f.write(json.dumps({
            "id": gcs_batch_job.name,
            "src": gcs_batch_job.src.gcs_uri,
            "dest": gcs_batch_job.dest.gcs_uri,
            "metadata": {
                "sample_ids": sample_ids,
            },
        }, indent=4))

    print("Batch submission created with name:", gcs_batch_job.name)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, help="Path to YAML model config file")
    parser.add_argument("--data_file", type=str, help="Path to the data file")
    parser.add_argument("--log_file", type=str, help="Path to the output log file")
    parser.add_argument("--gcs_bucket_name", type=str, help="GCS bucket name")
    parser.add_argument("--gcs_data_path", type=str, help="GS Path where the data file is uploaded to GCP")
    parser.add_argument("--gcs_output_path", type=str, help="GS Path where the output file is saved in GCP")
    parser.add_argument("--model", type=str, help="OpenAI model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Generation top_p")
    parser.add_argument("--max_tokens", type=int, default=128000, help="Max tokens accepted by the model")
    parser.add_argument("--estimate_cost", action="store_true", help="Estimate cost of the batch job")
    
    args, _ = parser.parse_known_args()
    
    yaml_args = {}
    if args.model_config:
        with open(args.model_config, "r") as f:
            model_yaml_args = yaml.safe_load(f) or {}
            assert model_yaml_args.pop("model_type") == "gcp", "Model type must be 'gcp' in the model config file."
            yaml_args.update(model_yaml_args)
    
    parser.set_defaults(**yaml_args)
    
    args = parser.parse_args()

    required_args = ["data_file", "log_file", "model", "max_tokens"]
    for arg in required_args:
        if getattr(args, arg) is None:
            raise ValueError(f"Missing required argument: --{arg}. Provide it in YAML or as a command-line argument.")

    for arg in ["gcs_data_path", "gcs_output_path"]:
        if getattr(args, arg) is None:
            raise ValueError(f"Missing required argument: --{arg}. Provide it in YAML or as a command-line argument.")

    log_dir = os.path.dirname(args.log_file)
    assert os.path.exists(log_dir), f"Log directory {log_dir} does not exist. Please create it before running the script."
        
    return args


def main():
    args = load_args()
    
    prepare_upload_submit_batch_file(
        data_file=args.data_file,
        log_file=args.log_file,
        model=args.model,
        gcs_bucket_name=args.gcs_bucket_name,
        gcs_data_path=args.gcs_data_path,
        gcs_output_path=args.gcs_output_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        estimate_cost=args.estimate_cost,
    )
    

if __name__ == "__main__":
    main()
