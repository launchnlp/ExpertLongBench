import os
import json
import argparse
import numpy as np
from scipy.stats import sem, bootstrap
from typing import List, Dict, Any
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score

def compute_performance_from_result_data(
    result_data: Dict[str, Dict[str, Dict[str, List[str]]]],
    checklist_key_list: List[str],
    poll_model_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    '''
        Aggregates the results stored in the result_data
    '''

    # final data structure to be used
    final_result_data = {
        'mean_precision': {'overall': 0, 'checklist': {}},
        'mean_recall': {'overall': 0, 'checklist': {}},
        'mean_f1_score': {'overall': 0, 'checklist': {}},
        'mean_binary_performance': {'overall': 0, 'checklist': {}},
        'mean_ternary_performance': {'overall': 0, 'checklist': {}},
        'poll_model_list': []
    }

    # iterating over the checklist_items
    for checklist_key in checklist_key_list:
        checklist_precision_matrix = []
        checklist_recall_matrix = []
        checklist_binary_performance_matrix = []
        checklist_ternary_performance_matrix = []

        # iterating over the models
        for model_name in poll_model_names:

            # iterating over the checklist keys in the model specific results
            precision = result_data[model_name]['precision'][checklist_key]
            recall = result_data[model_name]['recall'][checklist_key]
            binary_performance = result_data[model_name]['binary_performance'][checklist_key]
            ternary_performance = result_data[model_name]['ternary_performance'][checklist_key]

            # aggregating the performance
            checklist_precision_matrix.append(precision)
            checklist_recall_matrix.append(recall)
            checklist_binary_performance_matrix.append(binary_performance)
            checklist_ternary_performance_matrix.append(ternary_performance)

        # computing elementwise mean and max along the model axis using numpy
        checklist_precision_matrix = np.array(checklist_precision_matrix)
        mean_precision_matrix = np.mean(checklist_precision_matrix, axis=0)
        checklist_recall_matrix = np.array(checklist_recall_matrix)
        mean_recall_matrix = np.mean(checklist_recall_matrix, axis=0)
        checklist_binary_performance_matrix = np.array(checklist_binary_performance_matrix)
        mean_bp_matrix = np.mean(checklist_binary_performance_matrix, axis=0)
        checklist_ternary_performance_matrix = np.array(checklist_ternary_performance_matrix)
        mean_tp_matrix = np.mean(checklist_ternary_performance_matrix, axis=0)
        
        # updating the final result data
        final_result_data['mean_precision']['checklist'][checklist_key] = np.mean(mean_precision_matrix)
        final_result_data['mean_recall']['checklist'][checklist_key] = np.mean(mean_recall_matrix)
        final_result_data['mean_binary_performance']['checklist'][checklist_key] = np.mean(mean_bp_matrix)
        final_result_data['mean_ternary_performance']['checklist'][checklist_key] = np.mean(mean_tp_matrix)

    # iterating over the model names to compute the overall performance
    precision_matrix = []
    recall_matrix = []
    f1_score_matrix = []
    binary_performance_matrix = []
    ternary_performance_matrix = []
    for model_name in poll_model_names:
        precision = result_data[model_name]['precision']['overall']
        recall = result_data[model_name]['recall']['overall']
        f1_score = result_data[model_name]['f1_score']['overall']
        binary_performance = result_data[model_name]['binary_performance']['overall']
        ternary_performance = result_data[model_name]['ternary_performance']['overall']

        # aggregating the performance
        precision_matrix.append(precision)
        recall_matrix.append(recall)
        f1_score_matrix.append(f1_score)
        binary_performance_matrix.append(binary_performance)
        ternary_performance_matrix.append(ternary_performance)

    # computing the row-wise mean and the averaging the values
    precision_matrix = np.array(precision_matrix)
    mean_precision_matrix = np.mean(precision_matrix, axis=0)
    recall_matrix = np.array(recall_matrix)
    mean_recall_matrix = np.mean(recall_matrix, axis=0)
    f1_score_matrix = np.array(f1_score_matrix)
    mean_f1_score_matrix = np.mean(f1_score_matrix, axis=0)
    binary_performance_matrix = np.array(binary_performance_matrix)
    mean_bp_matrix = np.mean(binary_performance_matrix, axis=0)
    ternary_performance_matrix = np.array(ternary_performance_matrix)
    mean_tp_matrix = np.mean(ternary_performance_matrix, axis=0)
    
    # updating the final result data
    res = bootstrap((mean_precision_matrix,), np.mean, confidence_level=0.95, n_resamples=10000, method='percentile')
    final_result_data['mean_precision']['sem'] = res.standard_error
    final_result_data['mean_precision']['overall'] = np.mean(mean_precision_matrix)

    res = bootstrap((mean_recall_matrix,), np.mean, confidence_level=0.95, n_resamples=10000, method='percentile')
    final_result_data['mean_recall']['sem'] = res.standard_error
    final_result_data['mean_recall']['overall'] = np.mean(mean_recall_matrix)

    res = bootstrap((mean_f1_score_matrix,), np.mean, confidence_level=0.95, n_resamples=10000, method='percentile')
    final_result_data['mean_f1_score']['sem'] = res.standard_error
    final_result_data['mean_f1_score']['overall'] = np.mean(mean_f1_score_matrix)

    res = bootstrap((mean_bp_matrix,), np.mean, confidence_level=0.95, n_resamples=10000, method='percentile')
    final_result_data['mean_binary_performance']['sem'] = res.standard_error
    final_result_data['mean_binary_performance']['overall'] = np.mean(mean_bp_matrix)

    res = bootstrap((mean_tp_matrix,), np.mean, confidence_level=0.95, n_resamples=10000, method='percentile')
    final_result_data['mean_ternary_performance']['sem'] = res.standard_error
    final_result_data['mean_ternary_performance']['overall'] = np.mean(mean_tp_matrix)

    return final_result_data

def compute_agreement_from_result_data(
    result_data: Dict[str, Dict[str, Dict[str, List[str]]]],
    checklist_key_list: List[str],
    poll_model_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    '''
        Computes the agreement between the different models in binary and ternary performance
    '''

    # final data structure to be used
    final_result_data = {
        'binary_performance': {'overall': 0, 'checklist': {}},
        'ternary_performance': {'overall': 0, 'checklist': {}},
        'binary_agreement': {'overall': 0, 'checklist': {}},
        'ternary_agreement': {'overall': 0, 'checklist': {}},
        'poll_model_list': []
    }
    overall_checklist_binary_performance_matrix = []
    overall_checklist_ternary_performance_matrix = []


    # iterating over the checklist_items
    for checklist_key in checklist_key_list:
        checklist_binary_performance_matrix = []
        checklist_ternary_performance_matrix = []

        # iterating over the models
        for model_name in poll_model_names:

            # iterating over the checklist keys in the model specific results
            binary_performance = result_data[model_name]['binary_performance'][checklist_key]
            ternary_performance = result_data[model_name]['ternary_performance'][checklist_key]

            # aggregating the performance
            checklist_binary_performance_matrix.append(binary_performance)
            checklist_ternary_performance_matrix.append(ternary_performance)

        # taking the transpose of the matrix
        checklist_binary_performance_matrix = np.array(checklist_binary_performance_matrix).T
        checklist_ternary_performance_matrix = np.array(checklist_ternary_performance_matrix).T

        # converting the above matrices into int
        checklist_binary_performance_matrix = np.array(checklist_binary_performance_matrix, dtype=int)
        checklist_ternary_performance_matrix = np.array(checklist_ternary_performance_matrix * 2, dtype=int)

        # computing the agreement using cohen kappa
        cohen_kappa_binary_performance = cohen_kappa_score(checklist_binary_performance_matrix[:, 0], checklist_binary_performance_matrix[:, 1])
        cohen_kappa_ternary_performance = cohen_kappa_score(checklist_ternary_performance_matrix[:, 0], checklist_ternary_performance_matrix[:, 1])

        # computing the agreement as the number of samples in which the models agree / total number of samples
        binary_agreement = np.sum(checklist_binary_performance_matrix[:, 0] == checklist_binary_performance_matrix[:, 1]) / checklist_binary_performance_matrix.shape[0]
        ternary_agreement = np.sum(checklist_ternary_performance_matrix[:, 0] == checklist_ternary_performance_matrix[:, 1]) / checklist_ternary_performance_matrix.shape[0]

        # adding the binary and ternary performance to the overall checklist performance matrix
        overall_checklist_binary_performance_matrix.append(checklist_binary_performance_matrix)
        overall_checklist_ternary_performance_matrix.append(checklist_ternary_performance_matrix)

        # updating the final result data
        final_result_data['binary_performance']['checklist'][checklist_key] = cohen_kappa_binary_performance
        final_result_data['ternary_performance']['checklist'][checklist_key] = cohen_kappa_ternary_performance
        final_result_data['binary_agreement']['checklist'][checklist_key] = binary_agreement
        final_result_data['ternary_agreement']['checklist'][checklist_key] = ternary_agreement

    # taking the transpose of the overall checklist performance matrix
    overall_checklist_binary_performance_matrix = np.array(overall_checklist_binary_performance_matrix)
    overall_checklist_ternary_performance_matrix = np.array(overall_checklist_ternary_performance_matrix)

    # reshape it to be of shape (num_samples, num_models)
    overall_checklist_binary_performance_matrix = overall_checklist_binary_performance_matrix.reshape(-1, len(poll_model_names))
    overall_checklist_ternary_performance_matrix = overall_checklist_ternary_performance_matrix.reshape(-1, len(poll_model_names))

    # computing the overall results in final result data
    final_result_data['binary_performance']['overall'] = cohen_kappa_score(overall_checklist_binary_performance_matrix[:, 0], overall_checklist_binary_performance_matrix[:, 1])
    final_result_data['ternary_performance']['overall'] = cohen_kappa_score(overall_checklist_ternary_performance_matrix[:, 0], overall_checklist_ternary_performance_matrix[:, 1])
    final_result_data['binary_agreement']['overall'] = np.mean(list(final_result_data['binary_agreement']['checklist'].values()))
    final_result_data['ternary_agreement']['overall'] = np.mean(list(final_result_data['ternary_agreement']['checklist'].values()))

    # adding the poll model names to the final result data
    final_result_data['poll_model_list'] = poll_model_names
    return final_result_data


def checklist_comparison_performance(
    input_path: str,
    poll_model_names: List[str],
    output_path: str
) -> None:
    '''
        Aggregates the results stored in the input_path
    '''

    # loading the json file
    with open(input_path, 'r') as f:
        input_data = json.load(f)

    # extract all the checklist keys
    checklist_key_set = set()
    for element in input_data.values():
        element_results = element['results']
        for model_name in poll_model_names:
            model_specific_results = element_results[model_name]['performance']
            for checklist_key in model_specific_results.keys():
                checklist_key_set.add(checklist_key)
    checklist_key_list = list(checklist_key_set)

    # creating the data structure for result computation
    result_data = defaultdict(lambda: {
        'precision': {
            checklist_key: [] for checklist_key in checklist_key_list + ['overall']
        },
        'recall': {
            checklist_key: [] for checklist_key in checklist_key_list + ['overall']
        },
        'f1_score': {
            checklist_key: [] for checklist_key in checklist_key_list + ['overall']
        },
        'binary_performance': {
            checklist_key: [] for checklist_key in checklist_key_list + ['overall']
        },
        'ternary_performance': {
            checklist_key: [] for checklist_key in checklist_key_list + ['overall']
        },
    })

    # iterating over the input data and populating the result data
    for element in input_data.values():
        element_results = element['results']

        # iterating over the models
        for model_name in poll_model_names:
            model_specific_results = element_results[model_name]['performance']

            # creating instance specific lists
            instance_precision = []
            instance_recall = []
            instance_binary_performance = []
            instance_ternary_performance = []

            # iterating over the checklist keys in the model specific results
            for checklist_key in checklist_key_list:

                # check if the checklist key is in the model specific results
                if checklist_key not in model_specific_results:
                    continue

                obj = model_specific_results[checklist_key]
                binary_performance = (obj['precision'] * obj['recall'])
                ternary_performance = (obj['precision'] + obj['recall']) / 2
                f1_score = (2 * obj['precision'] * obj['recall']) / (obj['precision'] + obj['recall'] + 1e-10)

                # assinging the performance to the result data
                result_data[model_name]['precision'][checklist_key].append(obj['precision'])
                result_data[model_name]['recall'][checklist_key].append(obj['recall'])
                result_data[model_name]['binary_performance'][checklist_key].append(binary_performance)
                result_data[model_name]['ternary_performance'][checklist_key].append(ternary_performance)
                result_data[model_name]['f1_score'][checklist_key].append(f1_score)

                # adding the performance to the instance specific lists
                instance_precision.append(obj['precision'])
                instance_recall.append(obj['recall'])
                instance_binary_performance.append(binary_performance)
                instance_ternary_performance.append(ternary_performance)

            # computing the overall performance for the instance
            overall_precision = float(np.mean(instance_precision))
            overall_recall = float(np.mean(instance_recall))
            overall_f1_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + 1e-10)
            overall_binary_performance = float(np.mean(instance_binary_performance))
            overall_ternary_performance = float(np.mean(instance_ternary_performance))

            # assigning the overall performance to the result data
            result_data[model_name]['precision']['overall'].append(overall_precision)
            result_data[model_name]['recall']['overall'].append(overall_recall)
            result_data[model_name]['f1_score']['overall'].append(overall_f1_score)
            result_data[model_name]['binary_performance']['overall'].append(overall_binary_performance)
            result_data[model_name]['ternary_performance']['overall'].append(overall_ternary_performance)

    # computing the performance from the result data
    final_result_data = compute_performance_from_result_data(
        result_data=result_data,
        checklist_key_list=checklist_key_list,
        poll_model_names=poll_model_names
    )

    # create a directory for the output if it does not exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # adding the agreement result data to the final result data
    with open(output_path, 'w') as f:
        json.dump(final_result_data, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process the results stored in the data folder')
    parser.add_argument('--input_path', type=str, help='Path to the checklist parsed output JSON file where the evaluations are already stored.')
    parser.add_argument('--poll_model_names', type=str, nargs='+', default=['gpt-4o-2024-11-20'], help='List of model names whose evaluations would be aggregated. Default is gpt-4o-2024-11-20 only')
    parser.add_argument('--output_path', type=str, help='The path where the results will be saved')
    args = parser.parse_args()

    checklist_comparison_performance(
        input_path=args.input_path,
        poll_model_names=args.poll_model_names,
        output_path=args.output_path
    )