import os
import argparse
import jsonlines
import pandas as pd
from typing import Dict


def process_id(id_str: str) -> str:
    '''
        Function that processes the id string to remove the prefix
    '''
    id_str = id_str.split('-')[:2]
    return '-'.join(id_str)
    

def extract_checklist_from_excel(
    checklist_file: str,
    task_name: str
) -> Dict[str, str]:
    '''
        Function that extracts the checklist from the excel file
    '''
    
    # read the checklist file
    df = pd.read_excel(checklist_file, sheet_name=task_name)
    df = df.dropna(how='all', subset=[df.columns[0], df.columns[1]])
    data_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
    return data_dict


def checklist_extraction_request_creation(
    model_output: str,
    checklist_file: str,
    task_name: str,
    output_file: str
) -> None:
    '''
        Function that creates the checklist extraction request from the raw outputs
    '''

    # opening the checklist file
    checklist = extract_checklist_from_excel(checklist_file, task_name)
    checklist_keys = list(checklist.keys())

    # reading the model output jsonl file
    with open(model_output, 'r') as reader:
        model_output = []
        reader = jsonlines.Reader(reader)
        for item in reader:
            model_output.append(item)

    # iterating through the model output to create the final requests
    request_data = []
    for output_element in model_output:

        # iterating over the checklist
        for checklist_index, checklist_key in enumerate(checklist_keys):
            request_data.append({
                'id': output_element['id'] + '-' + str(checklist_index + 1),
                'system_prompt': checklist[checklist_key], 
                'sample': output_element['output'],
                'checklist_key': checklist_key,
                'max_output_tokens': 10000
            })


    # creating the output directory if it does not exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # writing the output to the file
    with open(output_file, 'w') as writer:
        writer = jsonlines.Writer(writer)
        for request in request_data:
            writer.write(request)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a checklist extraction request file')
    parser.add_argument('--model_output', type=str, help='Raw output that represents the inference from a particular model')
    parser.add_argument('--checklist_file', type=str, default='checklist_extraction/RefChecklistCreation-Prompt.xlsx', help='Path to the checklist prompts for each item')
    parser.add_argument('--task_name', type=str, default='T1LegalMDS', help='Task id associated with the raw output')
    parser.add_argument('--output_file', type=str, help='Path to where the request files will be saved')
    args = parser.parse_args()

    checklist_extraction_request_creation(
        model_output=args.model_output,
        checklist_file=args.checklist_file,
        task_name=args.task_name,
        output_file=args.output_file
    )


