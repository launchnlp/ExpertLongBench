import json
import argparse
import yaml
import os
from tqdm import tqdm
from dataclasses import dataclass, fields
from transformers import AutoTokenizer


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
    input_tokens = tokenizer(input_text, truncation=True, max_length=max_tokens, add_special_tokens=False)
    return tokenizer.decode(input_tokens["input_ids"], skip_special_tokens=True)


def run_model_offline(
    data_files,
    output_files,
    model,
    temperature=0.0,
    top_p=1.0,
    max_tokens=131072,
):
    import torch
    from vllm import LLM, SamplingParams
    model = LLM(
        model,
        tensor_parallel_size=torch.cuda.device_count() if "Qwen2.5-7B-Instruct" not in model else min(4, torch.cuda.device_count()),
        enable_prefix_caching=True,
        enforce_eager=True,
        trust_remote_code=True,
    )
    tokenizer = model.get_tokenizer()

    for data_file, output_file in zip(data_files, output_files):
        with open(data_file) as f:
            samples = [Example.from_json(json.loads(line)) for line in f]

        batch = []
        for sample in tqdm(samples, desc="Preprocessing samples"):
            input = truncate_input(sample.sample, max_tokens - sample.max_output_tokens - 1024, tokenizer)

            messages = [
                {"role": "system", "content": sample.system_prompt},
                {"role": "user", "content": input}
            ]

            batch.append((sample.id, messages, sample.max_output_tokens))

        # use batch if all samples have same max_output_tokens
        if len(set([x[2] for x in batch])) == 1:
            max_output_tokens = batch[0][2]

            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_output_tokens,
            )

            sample_ids = [x[0] for x in batch]
            batch = [x[1] for x in batch]

            outputs = model.chat(
                messages=batch,
                sampling_params=sampling_params,
                use_tqdm=True
            )

            with open(output_file, "w") as out_f:
                for i, output in enumerate(outputs):
                    output_text = output.outputs[0].text.strip()
                    out_f.write(json.dumps({
                        "id": sample_ids[i],
                        "output": output_text,
                    }) + "\n")

        else:
            with open(output_file, "w") as out_f:
                for sample_id, messages, max_output_tokens in tqdm(batch, desc="Generating outputs (non-batched)"):
                    sampling_params = SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_output_tokens,
                    )
                    chat_completion = model.chat(
                        messages,
                        sampling_params=sampling_params
                    )
                    output = chat_completion[0].outputs[0].text.strip()

                    out_f.write(json.dumps({
                        "id": sample_id,
                        "output": output,
                    }) + "\n")
        

def run_model_api(
    data_file,
    output_file,
    model,
    vllm_api_url=None,
    temperature=0.0,
    top_p=1.0,
    max_tokens=131072,
):
    from openai import OpenAI
    client = OpenAI(
        api_key="EMPTY",
        base_url=vllm_api_url,
        timeout=1200,
    )
    models = client.models.list()
    assert model == models.data[0].id, f"Model {model} not matched with vLLM API model {models.data[0].id}"

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    with open(data_file) as f:
        samples = [Example.from_json(json.loads(line)) for line in f]

    with open(output_file, "w") as out_f:
        for sample in tqdm(samples):
            input = truncate_input(sample.sample, max_tokens - sample.max_output_tokens - 1024, tokenizer)

            messages = [
                {"role": "system", "content": sample.system_prompt},
                {"role": "user", "content": input}
            ]

            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=sample.max_output_tokens,
                max_completion_tokens=sample.max_output_tokens,
            )
            output = chat_completion.choices[0].message.content

            out_f.write(json.dumps({
                "id": sample.id,
                "output": output,
            }) + "\n")


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, help="Path to YAML model config file")
    parser.add_argument("--data_file", type=str, help="Path to the data file", action="append")
    parser.add_argument("--output_file", type=str, help="Path to the output file", action="append")
    parser.add_argument("--model", type=str, help="Huggingface model name")
    parser.add_argument("--vllm_offline", action="store_true", default=False, help="Use vLLM offline prediciton")
    parser.add_argument("--vllm_api_url", type=str, help="vLLM API URL")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Generation top_p")
    parser.add_argument("--max_tokens", type=int, default=131072, help="Max tokens accepted by the model")
    
    args, _ = parser.parse_known_args()
    
    yaml_args = {}
    if args.model_config:
        with open(args.model_config, "r") as f:
            model_yaml_args = yaml.safe_load(f) or {}
            assert model_yaml_args.pop("model_type") == "huggingface", "Model type must be huggingface for this script"
            yaml_args.update(model_yaml_args)
    
    parser.set_defaults(**yaml_args)
    
    args = parser.parse_args()

    required_args = ["data_file", "output_file", "model", "max_tokens"]
    for arg in required_args:
        if getattr(args, arg) is None:
            raise ValueError(f"Missing required argument: --{arg}. Provide it in YAML or as a command-line argument.")
    
    required_non_empty_args = ["data_file", "output_file"]
    for arg in required_non_empty_args:
        if not getattr(args, arg):
            raise ValueError(f"Argument --{arg} cannot be empty. Provide a valid path.")
        
    assert len(args.data_file) == len(args.output_file), "Number of data files must match number of output files."
        
    if not args.vllm_offline and args.vllm_api_url is None:
        raise ValueError("If --vllm_offline is False, --vllm_api_url must be provided.")

    for output_file in args.output_file:
        output_dir = os.path.dirname(output_file)
        assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist. Please create it before running the script."

    return args


def main():
    args = load_args()
    
    if args.vllm_offline:
        run_model_offline(
            data_files=args.data_file,
            output_files=args.output_file,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
    else:
        run_model_api(
            data_file=args.data_file[0],
            output_file=args.output_file[0],
            model=args.model,
            vllm_api_url=args.vllm_api_url,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
    

if __name__ == "__main__":
    main()
