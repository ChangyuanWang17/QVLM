import argparse
import json
import os
import re
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--output-file', type=str)
    parser.add_argument('--output-result', type=str)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    return parser.parse_args()


def convert_caps(results):
    fakecaps = []
    for result in results:
        image_id = result['question_id']
        caption = result['text']
        fakecaps.append({"image_id": int(image_id), "caption": caption})
    return fakecaps


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))


if __name__ == "__main__":
    args = get_args()

    base_dir = args.base_dir
    split_indices = json.load(open(os.path.join(base_dir, "pid_splits.json")))[args.split]
    problems = json.load(open(os.path.join(base_dir, "problems.json")))
    predictions = [json.loads(line) for line in open(args.result_file)]
    predictions = {pred['question_id']: pred for pred in predictions}
    split_problems = {idx: problems[idx] for idx in split_indices}

    results = {'correct': [], 'incorrect': []}
    subject_results = {'NATcorrect': [], 'NATincorrect': [], 'SCOcorrect': [], 'SCOincorrect': [], 'LANcorrect': [], 'LANincorrect': []}
    context_results = {'TXTcorrect': [], 'TXTincorrect': [], 'IMGcorrect': [], 'IMGincorrect': [], 'NOcorrect': [], 'NOincorrect': []}
    grade_results = {'G1correct': [], 'G1incorrect': [], 'G7correct': [], 'G7incorrect': []}

    sqa_results = {}
    sqa_results['acc'] = None
    sqa_results['correct'] = None
    sqa_results['count'] = None
    sqa_results['results'] = {}
    sqa_results['outputs'] = {}

    for prob_id, prob in split_problems.items():
        if prob_id not in predictions:
            continue
        pred = predictions[prob_id]
        pred_text = pred['text']

        # pattern = re.compile(r'The answer is ([A-Z]).')
        # res = pattern.findall(pred_text)
        pattern = re.compile(r'ANSWER: ([A-Z]).')
        # pattern = re.compile(r'[A-Z]')
        res = pattern.findall(pred_text)
        if len(res) == 1:
            answer = res[0]  # 'A', 'B', ...
        else:
            answer = "FAILED"

        pred_idx = get_pred_idx(answer, prob['choices'], args.options)

        analysis = {
            'question_id': prob_id,
            'parsed_ans': answer,
            'ground_truth': args.options[prob['answer']],
            'question': pred['prompt'],
            'pred': pred_text,
            'is_multimodal': '<image>' in pred['prompt'],
        }

        sqa_results['results'][prob_id] = get_pred_idx(answer, prob['choices'], args.options)
        sqa_results['outputs'][prob_id] = pred_text

        if pred_idx == prob['answer']:
            results['correct'].append(analysis)
        else:
            results['incorrect'].append(analysis)

        # 计算其他
        # Subject
        subject_class = prob['subject']
        if subject_class == 'natural science':
            if pred_idx == prob['answer']:
                subject_results['NATcorrect'].append(analysis)
            else:
                subject_results['NATincorrect'].append(analysis)
        elif subject_class =='social science':
            if pred_idx == prob['answer']:
                subject_results['SCOcorrect'].append(analysis)
            else:
                subject_results['SCOincorrect'].append(analysis)
        else:
            if pred_idx == prob['answer']:
                subject_results['LANcorrect'].append(analysis)
            else:
                subject_results['LANincorrect'].append(analysis)

        # Context
        language_class = prob['hint']
        image_class = prob['image']
        # print(len(language_class), image_class==None)
        # 文本
        if len(language_class) != 0:
            if pred_idx == prob['answer']:
                context_results['TXTcorrect'].append(analysis)
            else:
                context_results['TXTincorrect'].append(analysis)
        if image_class!=None:
            if pred_idx == prob['answer']:
                context_results['IMGcorrect'].append(analysis)
            else:
                context_results['IMGincorrect'].append(analysis)
        if len(language_class) == 0 and image_class==None:
            if pred_idx == prob['answer']:
                context_results['NOcorrect'].append(analysis)
            else:
                context_results['NOincorrect'].append(analysis)

        # grade_results
        grade_class = prob['grade']
        if grade_class == 'grade1' or grade_class == 'grade2' or grade_class == 'grade3'or grade_class == 'grade4'or grade_class == 'grade5'or grade_class == 'grade6':
            if pred_idx == prob['answer']:
                grade_results['G1correct'].append(analysis)
            else:
                grade_results['G1incorrect'].append(analysis)
        elif grade_class == 'grade7' or grade_class == 'grade8' or grade_class == 'grade9'or grade_class == 'grade10'or grade_class == 'grade11'or grade_class == 'grade12':
            if pred_idx == prob['answer']:
                grade_results['G7correct'].append(analysis)
            else:
                grade_results['G7incorrect'].append(analysis)


    correct = len(results['correct'])
    total = len(results['correct']) + len(results['incorrect'])
    print(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%')

    sqa_results['acc'] = correct / total * 100
    sqa_results['correct'] = correct
    sqa_results['count'] = total

    # 计算其他
    # Subject
    NATcorrect = len(subject_results['NATcorrect'])
    NATtotal = len(subject_results['NATcorrect'])+len(subject_results['NATincorrect'])
    SCOcorrect = len(subject_results['SCOcorrect'])
    SCOtotal = len(subject_results['SCOcorrect'])+len(subject_results['SCOincorrect'])
    LANcorrect = len(subject_results['LANcorrect'])
    LANtotal = len(subject_results['LANcorrect'])+len(subject_results['LANincorrect'])
    print(f'NATAccuracy: {NATcorrect / NATtotal * 100:.2f}%')
    print(f'SCOAccuracy: {SCOcorrect / SCOtotal * 100:.2f}%')
    print(f'LANAccuracy: {LANcorrect / LANtotal * 100:.2f}%')
    sqa_results['NATacc'] = NATcorrect / NATtotal * 100
    sqa_results['SCOacc'] = SCOcorrect / SCOtotal * 100
    sqa_results['LANacc'] = LANcorrect / LANtotal * 100

    # Context
    TXTcorrect = len(context_results['TXTcorrect'])
    TXTtotal = len(context_results['TXTcorrect'])+len(context_results['TXTincorrect'])
    IMGcorrect = len(context_results['IMGcorrect'])
    IMGtotal = len(context_results['IMGcorrect'])+len(context_results['IMGincorrect'])
    NOcorrect = len(context_results['NOcorrect'])
    NOtotal = len(context_results['NOcorrect'])+len(context_results['NOincorrect'])
    print(f'TXTAccuracy: {TXTcorrect / TXTtotal * 100:.2f}%')
    print(f'IMGAccuracy: {IMGcorrect / IMGtotal * 100:.2f}%')
    print(f'NOAccuracy: {NOcorrect / NOtotal * 100:.2f}%')
    sqa_results['TXTacc'] = TXTcorrect / TXTtotal * 100
    sqa_results['IMGacc'] = IMGcorrect / IMGtotal * 100
    sqa_results['NOacc'] = NOcorrect / NOtotal * 100

    # grade
    G1correct = len(grade_results['G1correct'])
    G1total = len(grade_results['G1correct'])+len(grade_results['G1incorrect'])
    G7correct = len(grade_results['G7correct'])
    G7total = len(grade_results['G7correct'])+len(grade_results['G7incorrect'])
    print(f'G1Accuracy: {G1correct / G1total * 100:.2f}%')
    print(f'G7Accuracy: {G7correct / G7total * 100:.2f}%')
    sqa_results['G1acc'] = G1correct / G1total * 100
    sqa_results['G7acc'] = G7correct / G7total * 100


    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    with open(args.output_result, 'w') as f:
        json.dump(sqa_results, f, indent=2)
