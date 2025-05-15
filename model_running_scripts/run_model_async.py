import json
import argparse
import yaml
import os
from tqdm import tqdm
from dataclasses import dataclass, fields
from transformers import AutoTokenizer
import asyncio


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


async def run_single_request(client, model, sample, tokenizer, temperature, top_p, max_tokens):
    from openai import APITimeoutError
    input_text = truncate_input(sample.sample, max_tokens - sample.max_output_tokens - 1024, tokenizer)

    messages = [
        {"role": "system", "content": sample.system_prompt},
        {"role": "user", "content": input_text}
    ]

    try:
        chat_completion = await client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=sample.max_output_tokens,
            max_completion_tokens=sample.max_output_tokens,
        )
        output = chat_completion.choices[0].message.content
    except APITimeoutError as e:
        return {"id": sample.id, "output": "TIMEOUT"}

    return {"id": sample.id, "output": output}


async def run_model_api(
    data_file,
    output_file,
    model,
    vllm_api_url=None,
    temperature=0.0,
    top_p=1.0,
    max_tokens=131072,
):
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url=vllm_api_url,
        timeout=3600,
    )

    models = await client.models.list()
    assert model == models.data[0].id, f"Model {model} not matched with vLLM API model {models.data[0].id}"

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

    with open(data_file) as f:
        samples = [Example.from_json(json.loads(line)) for line in f]

    results = []
    sem = asyncio.Semaphore(20)  # Control concurrency level

    async def sem_task(sample):
        async with sem:
            return await run_single_request(client, model, sample, tokenizer, temperature, top_p, max_tokens)

    for fut in tqdm(asyncio.as_completed([sem_task(s) for s in samples]), total=len(samples)):
        result = await fut
        results.append(result)

    with open(output_file, "w") as out_f:
        for item in results:
            out_f.write(json.dumps(item) + "\n")


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", type=str, help="Path to YAML model config file")
    parser.add_argument("--data_file", type=str, help="Path to the data file")
    parser.add_argument("--output_file", type=str, help="Path to the output file")
    parser.add_argument("--model", type=str, help="Huggingface model name")
    parser.add_argument("--vllm_offline", action="store_true", default=False, help="Use vLLM offline prediction")
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

    if not args.vllm_offline and args.vllm_api_url is None:
        raise ValueError("If --vllm_offline is False, --vllm_api_url must be provided.")

    output_dir = os.path.dirname(args.output_file)
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist. Please create it before running the script."

    return args


async def main():
    args = load_args()

    if args.vllm_offline:
        raise NotImplementedError("vLLM offline prediction is not implemented in this script.")
    else:
        await run_model_api(
            data_file=args.data_file,
            output_file=args.output_file,
            model=args.model,
            vllm_api_url=args.vllm_api_url,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )


if __name__ == "__main__":
    asyncio.run(main())
