import json
import argparse
import yaml
import tempfile
import os
from dataclasses import dataclass, fields
from openai import OpenAI
import tiktoken


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


def truncate_input(input_text, max_tokens, tokenizer):
    input_tokens = tokenizer.encode(input_text)
    return tokenizer.decode(input_tokens[:max_tokens])


def prepare_upload_submit_batch_file(
    data_file,
    log_file,
    model,
    temperature=0.0,
    top_p=1.0,
    max_tokens=131072,
    estimate_cost=False,
):
    client = OpenAI()

    tokenizer = tiktoken.encoding_for_model(model)

    with open(data_file) as f:
        samples = [Example.from_json(json.loads(line)) for line in f]
    
    if estimate_cost:
        input_tokens = 0
        output_tokens = 0
        for sample in samples:
            input = truncate_input(sample.sample, max_tokens - sample.max_output_tokens - 1024, tokenizer)
            input_tokens += len(tokenizer.encode(input))
            output_tokens += sample.max_output_tokens
        
        if model == "gpt-4o-2024-11-20":
            cost_per_1k_input = 2.5 / 1000
            cost_per_1k_output = 10 / 1000
        elif model == "gpt-4o-mini-2024-07-18":
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

    with tempfile.NamedTemporaryFile("w") as temp_file:
        for sample in samples:
            input = truncate_input(sample.sample, max_tokens - sample.max_output_tokens - 1024, tokenizer)

            batch_sample = {
                "custom_id": sample.id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": sample.system_prompt},
                        {"role": "user", "content": input}
                    ],
                    "max_tokens": sample.max_output_tokens,
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
            temp_file.write(json.dumps(batch_sample) + "\n")
        temp_file.flush()

        batch_input_file = client.files.create(
            file=open(temp_file.name, "rb"),
            purpose="batch"
        )

    print("Batch input file created with ID:", batch_input_file.id)

    batch_input_file_id = batch_input_file.id

    batch_submission = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"Batch submission for {data_file}",
        }
    )

    with open(log_file, "w") as f:
        f.write(json.dumps({
            "id": batch_submission.id,
            "input_file_id": batch_submission.input_file_id,
            "metadata": batch_submission.metadata,
        }, indent=4))

    print("Batch submission created with ID:", batch_submission.id)


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, help="Path to YAML model config file")
    parser.add_argument("--data_file", type=str, help="Path to the data file")
    parser.add_argument("--log_file", type=str, help="Path to the output log file")
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
            assert model_yaml_args.pop("model_type") == "openai", "Model type must be 'openai' in the model config file."
            yaml_args.update(model_yaml_args)
    
    parser.set_defaults(**yaml_args)
    
    args = parser.parse_args()

    required_args = ["data_file", "log_file", "model", "max_tokens"]
    for arg in required_args:
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
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        estimate_cost=args.estimate_cost
    )
    

if __name__ == "__main__":
    main()
