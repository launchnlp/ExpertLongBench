import json
import argparse
import jsonlines
from collections import defaultdict
from typing import List, Dict, Tuple

def parse_id(
    id_string: str
) -> Tuple[str, str]:
    '''
        Breaks the id_string into the id of the example and the id of the checklist item
    '''

    # Split the string by '-'
    parts = id_string.split('-')
    example_id = '-'.join(parts[:2])
    checklist_id = '-'.join(parts[2:])
    return example_id, checklist_id

def create_integrated_data_schema(
    parsed_request_data: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    '''
        Creates the integrated data schema when the checklist associated with the examples are variable
    '''
    integrated_data = dict()
    for example_id in parsed_request_data.keys():
        integrated_data[example_id] = dict()
        for checklist_id in parsed_request_data[example_id].keys():
            checklist_key = parsed_request_data[example_id][checklist_id]['checklist_key']
            integrated_data[example_id][checklist_key] = {
                'precision': 0.0,
                'recall': 0.0
            }
    return integrated_data

def parser_request_data(
    request_data_list: List[Dict[str, str]]
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    '''
        From the request data, it maps the example id to a dictionary which further maps the checklist id to the actual checklist item
    '''
    
    # iterating over the request data list
    parsed_request_data = defaultdict(dict)
    checklist_key_list = set()
    for element in request_data_list:

        # parse the id
        example_id, checklist_id = parse_id(element['id'])

        # create entry
        parsed_request_data[example_id][checklist_id] = {
            'checklist_key': element['checklist_key'],
            'value_forcing': element['value_forcing']
        }
        checklist_key_list.add(element['checklist_key'])

    return parsed_request_data, list(checklist_key_list)


def integrate_request_evaluation_data(
    parsed_request_data: Dict[str, Dict[str, str]],
    evaluation_data_list: List[Dict[str, str]],
    checklist_key_list: List[str],
    integrated_data: Dict[str, Dict[str, Dict[str, float]]] = None
) -> Dict[str, Dict[str, Dict[str, float]]]:
    '''
        Integrates the request data and the evaluation data
    '''

    # creating the final data structure
    if integrated_data is None:
        integrated_data = defaultdict(lambda: {
            key: {'precision': 0.0, 'recall': 0.0} for key in checklist_key_list
        })

    # iterating over the evaluation data
    for element in evaluation_data_list:

        # parse the id
        example_id, checklist_id = parse_id(element['id'])

        # extract the checklist key from the request data
        checklist_key = parsed_request_data[example_id][checklist_id]['checklist_key']
        value_forcing = parsed_request_data[example_id][checklist_id]['value_forcing']

        # update the integrated data
        if 'reverse' in element['id']:
            if value_forcing is None:
                integrated_data[example_id][checklist_key]['precision'] = 1.0 if 'yes' in element['output'].lower() else 0.0
            else:
                integrated_data[example_id][checklist_key]['precision'] = value_forcing
        else:
            if value_forcing is None:
                integrated_data[example_id][checklist_key]['recall'] = 1.0 if 'yes' in element['output'].lower() else 0.0
            else:
                integrated_data[example_id][checklist_key]['recall'] = value_forcing

    return integrated_data

def checklist_comparison_mapper(
    input_path: str,
    request_path: str,
    evaluation_path: str,
    checklist_comparator: str,
    checklist_gt_source: str,
    variable_checklist: bool = False
) -> None:
    '''
        Maps the checklist evaluation results to the original dataset
    '''

    # loading the json and jsonlines files (convert jsonlines to a list of dicts)
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    with open(request_path, 'r') as f:
        request_data = jsonlines.Reader(f)
        request_data_list = []
        for item in request_data:
            request_data_list.append(item)
    with open(evaluation_path, 'r') as f:
        evaluation_data = jsonlines.Reader(f)
        evaluation_data_list = []
        for item in evaluation_data:
            evaluation_data_list.append(item)
            
    # parsing the request data
    parsed_request_data, checklist_key_list = parser_request_data(request_data_list)

    # creating a structure for integrated_data if variable_checklist
    integrated_data = None
    if variable_checklist:
        integrated_data = create_integrated_data_schema(parsed_request_data)

    # integrating the request data and the evaluation data
    integrated_data = integrate_request_evaluation_data(
        parsed_request_data,
        evaluation_data_list,
        checklist_key_list,
        integrated_data = integrated_data
    )

    # adding the data back to the input data
    for example_id in input_data.keys():

        # get the input element
        input_element = input_data[example_id]

        # create a results key if it doesn't exist
        if 'results' not in input_element:
            input_element['results'] = {}

        # assign the results
        input_element['results'][checklist_comparator] = {
            'performance': integrated_data[example_id],
            'reference_mapped_checklist_source': checklist_gt_source
        }
    
    with open(input_path, 'w') as f:
        json.dump(input_data, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Maps the checklist evaluation results to the original dataset.')
    parser.add_argument('--input_path', type=str, help='Path to the input JSON file where the evaluation will be saved. This is the parsed output from the checklist extraction')
    parser.add_argument('--request_path', type=str, help='Path to the request JSONL file that contains checklist evaluation requests')
    parser.add_argument('--evaluation_path', type=str, help='Path to the output JSONL file that contains checklist evaluation results')
    parser.add_argument('--checklist_comparator', type=str, default='gpt-4o-2024-11-20', help='The model used for checklist evaluation')
    parser.add_argument('--checklist_gt_source', type=str, default='gpt-4o-2024-11-20', help='The model used for the ground truth checklist source')
    parser.add_argument('--variable_checklist', action='store_true', help='If set, the checklist is variable and the evaluation will be saved in a different format.')
    args = parser.parse_args()

    checklist_comparison_mapper(
        input_path=args.input_path,
        request_path=args.request_path,
        evaluation_path=args.evaluation_path,
        checklist_comparator=args.checklist_comparator,
        checklist_gt_source=args.checklist_gt_source,
        variable_checklist=args.variable_checklist
    )