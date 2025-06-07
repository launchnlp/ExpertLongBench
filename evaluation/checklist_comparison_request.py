'''
    This will create a request file that can be used for comparing the checklists
'''

import os
import json
import argparse
import jsonlines
from typing import Dict, Any

def create_system_prompt(
    prompt_dict: Dict[str, Any]
) -> str:
    '''
        Converts the prompt dictionary to a string
    '''
    
    # creating the few shot prompt
    few_shot_prompt_str = ''
    for example in prompt_dict['examples']:
        example_str = 'Model Answer: {}\nReference Answer: {}\nCorrect: {}\n----\n'.format(
            example['answer'],
            example['reference'],
            example['label']
        )
        few_shot_prompt_str += example_str

    # creating the final system prompt
    system_prompt = '{}\nStudy the following examples as they will be very informative for how to do the task.\n\n{}'.format(
        prompt_dict['instruction'],
        few_shot_prompt_str
    )
    return system_prompt

def checklist_comparison_request_creation(
    reference_file: str,
    inference_file: str,
    prompt_file: str,
    output_file: str
) -> None:
    '''
        For each datapoint in reference_file and inference_file, creates comparison requests for each checklist entry
    '''

    # loading the reference, inference and prompt files
    with open(reference_file, 'r') as f:
        reference_data = json.load(f)
    with open(inference_file, 'r') as f:
        inference_data = json.load(f)
    with open(prompt_file, 'r') as f:
        prompt_data = json.load(f)

    # making directory if it does not exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # creating the system prompt
    system_prompt = create_system_prompt(prompt_data)

    # creating the request file
    request_data = []
    for key in reference_data.keys():

        # getting the reference and inference data
        if key not in inference_data:
            print(f'Key {key} not found in inference data')
            continue
        reference_checklist = reference_data[key]['checklist']
        inference_checklist = inference_data[key]['checklist']

        # iterating over each element in the checklist
        for index, checklist_key in enumerate(reference_checklist.keys()):
            if checklist_key not in inference_checklist:
                print(f'Key {checklist_key} not found in inference data for inference file {inference_file}')
                continue

            # checking whether the value has to be forced
            value_forcing = None
            if reference_checklist[checklist_key] == 'N/A' or inference_checklist[checklist_key] == 'N/A':
                value_forcing = 1.0 if reference_checklist[checklist_key] == inference_checklist[checklist_key] else 0.0

            # creating the request
            id_key = key + '-' + str(index + 1)
            sample = 'Model Answer: {}\nReference Answer: {}\nCorrect: '.format(
                inference_checklist[checklist_key],
                reference_checklist[checklist_key],
            )
            request_data.append({
                'id': id_key,
                'system_prompt': system_prompt,
                'sample': sample,
                'max_output_tokens': 5,
                'checklist_key': checklist_key,
                'value_forcing': value_forcing
            })

            # creating reverse request
            id_key = key + '-' + str(index + 1) + '-reverse'
            sample = 'Model Answer: {}\nReference Answer: {}\nCorrect: '.format(
                reference_checklist[checklist_key],
                inference_checklist[checklist_key],
            )
            request_data.append({
                'id': id_key,
                'system_prompt': system_prompt,
                'sample': sample,
                'max_output_tokens': 5,
                'checklist_key': checklist_key,
                'value_forcing': value_forcing
            })

    # writing the request data to a jsonl file
    with jsonlines.open(output_file, 'w') as writer:
        for request in request_data:
            writer.write(request)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a checklist comparison request file')
    parser.add_argument('--reference_file', type=str, help='The path to the reference file which stores the checklist items for each sample id')
    parser.add_argument('--inference_file', type=str, default='The path to the inference file which stores the checklist items for each sample id')
    parser.add_argument('--prompt_file', type=str, default='evaluation/prompt_binary.json')
    parser.add_argument('--output_file', type=str, help='The path where the request file will be saved which can be used to generate inference for each checklist item for each sample id')
    args = parser.parse_args()

    # creating the checklist comparison request file
    checklist_comparison_request_creation(
        reference_file=args.reference_file,
        inference_file=args.inference_file,
        prompt_file=args.prompt_file,
        output_file=args.output_file
    )
