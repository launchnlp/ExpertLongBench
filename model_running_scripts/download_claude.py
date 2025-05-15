import json
import argparse
import os
import fsspec
from google import genai
from google.cloud import storage
from google.genai.types import JobState


def download_batch(
    log_file,
    output_file,
    cost_file,
):
    client = genai.Client(
        vertexai=True,
        project="jieruan-gcp-bd84",
        location="us-east5",
    )

    with open(log_file, "r") as f:
        batch_log = json.load(f)
        batch_id = batch_log["id"]
        sample_ids = batch_log["metadata"]["sample_ids"]

    batch = client.batches.get(name=batch_id)
    if batch.state != JobState.JOB_STATE_SUCCEEDED:
        print(f"Batch job {batch_id} not completed or failed.")
        print(batch)
        return

    batch_dest = batch.dest.gcs_uri

    fs = fsspec.filesystem("gcs")
    file_paths = fs.glob(f"{batch_dest}/*/predictions.jsonl")
    if not file_paths:
        print(f"No output files found in {batch_dest}.")
        return
    if len(file_paths) > 1:
        # get the lastest file
        # the name is "prediction-model-{timestamp}/predictions.jsonl"
        file_paths.sort()
        print(f"Found multiple output files: {file_paths}. Using the latest one: {file_paths[-1]}.")

    outputs = []
    usage_metadatas = []
    with fs.open(file_paths[-1], "r") as f:
        for line in f:
            prediction = json.loads(line)
            outputs.append({
                "id": prediction["custom_id"],
                "output": prediction["response"]["content"][0]["text"],
            })
            usage_metadatas.append({
                "id": prediction["custom_id"],
                "usage_metadata": prediction["response"]["usage"],
            })

    with open(output_file, "w") as f:
        for output in outputs:
            f.write(json.dumps(output) + "\n")
    with open(cost_file, "w") as f:
        for usage_metadata in usage_metadatas:
            f.write(json.dumps(usage_metadata) + "\n")
    print("Batch output downloaded successfully.")
    

def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, help="Path to the output log file", required=True)
    parser.add_argument("--output_file", type=str, help="Path to the output file", required=True)
    parser.add_argument("--cost_file", type=str, help="Path to the cost file", required=True)
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
    )
    

if __name__ == "__main__":
    main()
