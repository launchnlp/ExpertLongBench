import json
import argparse
import os
from openai import OpenAI


def download_batch(
    log_file,
    output_file,
    cost_file,
    overwrite=False,
):
    if os.path.exists(output_file) and not overwrite:
        print(f"Output file {output_file} already exists. Use --overwrite to overwrite it.")
        return

    client = OpenAI()

    with open(log_file, "r") as f:
        batch_log = json.load(f)
        batch_id = batch_log["id"]

    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        print("Batch is not completed.")
        print(batch)
        return
    
    if batch.request_counts.failed == 0:
        output_file_id = batch.output_file_id
        file_response = client.files.content(output_file_id)

        outputs = []
        usage_metadatas = []
        for line in file_response.text.strip().split("\n"):
            output = json.loads(line)
            output_id = output["custom_id"]
            response = output["response"]["body"]["choices"][0]["message"]["content"]
            outputs.append({
                "id": output_id,
                "output": response,
            })
            usage_metadatas.append({
                "id": output_id,
                "usage_metadata": output["response"]["body"]["usage"],
            })

        with open(output_file, "w") as f:
            for output in outputs:
                f.write(json.dumps(output) + "\n")
        with open(cost_file, "w") as f:
            for usage_metadata in usage_metadatas:
                f.write(json.dumps(usage_metadata) + "\n")
        print("Batch output downloaded successfully.")
    else:
        error_file_id = batch.error_file_id
        file_response = client.files.content(error_file_id)

        with open("error_file.jsonl", "w") as f:
            f.write(file_response.text)
        print("Batch failed with errors. Check error_file.jsonl for details.")


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, help="Path to the output log file", required=True)
    parser.add_argument("--output_file", type=str, help="Path to the output file", required=True)
    parser.add_argument("--cost_file", type=str, help="Path to the cost file", required=True)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it exists")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_file)
    assert os.path.exists(output_dir), f"Output directory {output_dir} does not exist. Please create it before running the script."

    return args


def main():
    args = load_args()
    
    download_batch(
        log_file=args.log_file,
        output_file=args.output_file,
        cost_file=args.cost_file,
        overwrite=args.overwrite,
    )
    

if __name__ == "__main__":
    main()
