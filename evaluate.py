import json
import argparse
import os
import sys
import random

def evaluate_performance(data, dataset_name):
    true_count = 0
    false_count = 0
    
    total_records = 0

    for i, record in enumerate(data):
        total_records += 1
        predicted_answer = record.get('predicted_choice', '').strip().lower()
        actual_answer = record.get('answer', '').strip().lower()
        
        if dataset_name in ['ProntoQA', 'ProofWriter', 'FOLIO']:
            try:
                if record['predicted_answer'] == "No final answer found in the text.":
                    predicted_answer = record['original_answer']
                    if predicted_answer == "true":
                        predicted_answer = 'a'
                    elif predicted_answer == "false":
                        predicted_answer = 'b'
                    elif predicted_answer == "unknown":
                        predicted_answer = 'c'
            except:
                pass        
        
        if dataset_name == "LogicalDeduction":
            valid_options = ['a', 'b', 'c', 'd', 'e']
        elif dataset_name == "AR-LSAT":
            valid_options = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
        
        if dataset_name in ['LogicalDeduction', 'AR-LSAT']:
            try:
                if record['predicted_answer'] == "No final answer found in the text.":
                    if 'original_answer' in record:
                        predicted_answer = record['original_answer']
                        if record['original_answer'] == "No final answer found in the text.":
                            predicted_answer = random.choice(valid_options)
            except:
                pass

        if predicted_answer == actual_answer:
            true_count += 1

    accuracy = (true_count / total_records) * 100 if total_records > 0 else 0
    false_count = total_records - true_count
    return true_count, false_count, total_records, accuracy

def compute_f1_score(true_positive, false_positive, false_negative):
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def evaluate(result_path):
    print("Result file: ", result_path)
    with open(result_path, 'r') as f:
        data = json.load(f)
    true_count, false_count, total_records, accuracy = evaluate_performance(data, args.dataset_name)
    print(f"Total records: {total_records}")
    print(f"Correctly predicted 'true': {true_count}")
    print(f"Correctly predicted 'false': {false_count}")
    print(f"Accuracy: {accuracy:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--mode', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--verification', type=str, default='False')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if args.verification == 'True':
        result_file = os.path.join('./verified_results', f'{args.mode}_{args.dataset_name}_{args.split}_{args.model_name}_verified.json')
    else:
        result_file = os.path.join(args.result_path, f'{args.mode}_{args.dataset_name}_{args.split}_{args.model_name}.json')
    evaluate(result_file)
