import json
import fire
import yaml


def create_task_output_requests(
    data_file: str,
    task_config: str,
    output_file: str,
):
    with open(data_file, "r") as f:
        samples = [json.loads(line) for line in f]

    with open(task_config, "r") as f:
        task_config = yaml.safe_load(f)

    task_output_requests = []
    for sample in samples:
        task_output_request = {
            "id": sample["id"],
            "system_prompt": task_config["system_prompt"],
            "sample": sample["input"],
            "max_output_tokens": task_config["max_output_tokens"],
        }
        task_output_requests.append(task_output_request)

    with open(output_file, "w") as f:
        for request in task_output_requests:
            f.write(json.dumps(request) + "\n")


if __name__ == "__main__":
    fire.Fire(create_task_output_requests)