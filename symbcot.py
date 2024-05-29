import json
import os
from tqdm import tqdm
from utils import OpenAIModel
import argparse
import re
import sys

class GPT3_Reasoning_Graph_Baseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.mode = args.mode
        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
            
    def load_in_context_examples_trans(self):
        file_path = os.path.join('./prompts', self.dataset_name, 'translation.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples

    def load_in_context_examples_plan(self):
        file_path = os.path.join('./prompts', self.dataset_name, 'plan_generation.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_in_context_examples_solve(self):
        file_path = os.path.join('./prompts', self.dataset_name, 'solver.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_raw_dataset(self, split):
        with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
            raw_dataset = json.load(f)
        return raw_dataset
    
    def index_context(self, context):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
        formatted_context = enumerate(sentences, start=1)
        indexed_sentences = '\n'.join([f"{index}: {sentence}" for index, sentence in formatted_context])
        return str(indexed_sentences)

    def construct_prompt_a(self, record, in_context_examples_trans):
        full_prompt = in_context_examples_trans
        context = record['context']
        question = record['question'].strip()
        full_prompt = full_prompt.replace('[[PROBLEM]]', context)
        full_prompt = full_prompt.replace('[[QUESTION]]', question)
        return full_prompt

    def construct_prompt_b(self, record, responses_a, in_context_examples_plan):
        full_prompt = in_context_examples_plan
        full_prompt = full_prompt.replace('[[CONTEXT]]', responses_a)
        return full_prompt

    def construct_prompt_c(self, responses_a, responses_b, in_context_examples_solve):
        full_prompt = in_context_examples_solve
        plan = responses_b
        full_prompt = full_prompt.replace('[[CONTEXT]]', responses_a)
        full_prompt = full_prompt.replace('[[PLAN]]', plan)
        return full_prompt

    def post_process_a(self, response_a):
        response_a = str(response_a)
        context_start = response_a.find('"context":') + 10
        context_end = response_a.find('",\n"Question"')
        context = response_a[context_start:context_end].strip()
        question_start = response_a.find('"Question":') + 11
        question_end = response_a[question_start:].find('"}') + question_start
        question = response_a[question_start:question_end].strip()
        return context, question
    
    def post_process_c(self, response_c):
        pattern_bracket = r"Final answer: \{([A-E])\}"
        match = re.search(pattern_bracket, response_c)
        if match:
            answers =  match.group(1)
            return answers
        pattern_direct = r'\{(\w+)\}'
        match = re.search(pattern_direct, response_c, re.IGNORECASE)
        if match:
            return match.group(1).lower()
        return "No final answer found in the text."

    
    def final_process(self, final_answer):
        final_answer = final_answer.lower()
        if final_answer == "true":
            final_answer = 'A'
        elif final_answer == "false":
            final_answer = 'B'
        elif final_answer == "unknown":
            final_answer = 'C'
        else:
            final_answer = "No final answer found in the text."  
        return final_answer
    
    def reasoning_graph_generation(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        # load in-context examples
        in_context_examples_trans = self.load_in_context_examples_trans()
        in_context_examples_plan = self.load_in_context_examples_plan()
        in_context_examples_solve = self.load_in_context_examples_solve()
        
        outputs = []
        error_output = []

        for example in tqdm(raw_dataset):
            question = example['question']
            try:
                print("Translating...")
                prompts_a = self.construct_prompt_a(example, in_context_examples_trans)
                responses_a, _ = self.openai_api.generate(prompts_a)
                
                print("Planning...")
                prompts_b = self.construct_prompt_b(example, responses_a, in_context_examples_plan)
                responses_b, _ = self.openai_api.generate(prompts_b)
                
                print("Solving...")
                prompts_c = self.construct_prompt_c(responses_a, responses_b, in_context_examples_solve)
                responses_c, finish_reason = self.openai_api.generate(prompts_c)

                final_answer = self.post_process_c(responses_c)
                final_choice = self.final_process(final_answer)
                
                output = {'id': example['id'], 
                        'question': question, 
                        'answer': example['answer'], 
                        'predicted_answer': final_answer,
                        'predicted_choice': final_choice,
                        'context': responses_a,
                        'plan': responses_b,
                        'execution': responses_c,
                        'finish_reason': finish_reason}
                print(output)
                outputs.append(output)
                
                with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
                    json.dump(outputs, f, indent=2, ensure_ascii=False)                    
                
            except Exception as e:
                print('Error in generating example: ', example['id'])
                print(e)
                error = {'id': example['id']}
                error_output.append(error)
                try:
                    with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_error.json'), 'w') as f:
                            json.dump(error_output, f, indent=2, ensure_ascii=False)         
                except:
                    print("Error in saving error output")
                continue
                            
        # save outputs        
        with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def batch_reasoning_graph_generation(self, batch_size=10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        # load in-context examples
        in_context_examples_trans = self.load_in_context_examples_trans()
        in_context_examples_plan = self.load_in_context_examples_plan()
        in_context_examples_solve = self.load_in_context_examples_solve()
                        
        outputs = []
        
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            try:
                print("Translating...")
                batch_translation = self.openai_api.batch_generate([self.construct_prompt_a(example, in_context_examples_trans) for example in chunk])
                
                print("Planning...")
                batch_plan = self.openai_api.batch_generate([self.construct_prompt_b(example, responses_a, in_context_examples_plan) for example, responses_a in zip(chunk, batch_translation)])
                
                print("Solving...")
                batch_outputs = self.openai_api.batch_generate([self.construct_prompt_c(responses_a, responses_b, in_context_examples_solve) for responses_a, responses_b in zip(batch_translation, batch_plan)])
                
                for sample, translation, plan, output in zip(chunk, batch_translation, batch_plan, batch_outputs):
                    dict_output = self.update_answer(sample, translation, plan, output)
                    outputs.append(dict_output)
                    
                with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
                    json.dump(outputs, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print("Error in batch generation: ", e)
                for sample in chunk:
                    try:
                        print("Translating...")
                        prompts_a = self.construct_prompt_a(sample, in_context_examples_trans)
                        responses_a, _ = self.openai_api.generate(prompts_a)
                        
                        print("Planning...")
                        prompts_b = self.construct_prompt_b(sample, responses_a, in_context_examples_plan)
                        responses_b, _ = self.openai_api.generate(prompts_b)
                        
                        print("Solving...")
                        prompts_c = self.construct_prompt_c(responses_a, responses_b, in_context_examples_solve)
                        output, _ = self.openai_api.generate(prompts_c)
                        
                        dict_output = self.update_answer(sample, responses_a, responses_b, output)
                        outputs.append(dict_output)
                        
                        with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
                            json.dump(outputs, f, indent=2, ensure_ascii=False)
                            
                    except Exception as e:
                        print('Error in generating example: ', sample['id'] , e)
                            

        with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
    
    def update_answer(self, sample, translation, plan, output):
        final_answer = self.post_process_c(output)
        final_choice = self.final_process(final_answer)
        dict_output = {'id': sample['id'],
                       'question': sample['question'],
                       'original_context': sample['context'],
                       'context': translation,
                       'plan': plan,
                       'execution': output,
                       'predicted_answer': final_answer, 
                       'answer': sample['answer'],
                       'predicted_choice': final_choice}
        return dict_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--save_path', type=str, default='./results')
    parser.add_argument('--demonstration_path', type=str, default='./icl_examples')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--max_new_tokens', type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gpt3_problem_reduction = GPT3_Reasoning_Graph_Baseline(args)
    # gpt3_problem_reduction.reasoning_graph_generation()
    gpt3_problem_reduction.batch_reasoning_graph_generation(batch_size=1)


