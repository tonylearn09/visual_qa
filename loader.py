import argparse
import os
import numpy as np
import re
import pickle
import h5py

try:
    import json
except ImportError:
    import simplejson as json 

def load_qas(args):
    # Already prepared data, i.e. run it before
    qa_data = os.path.join(args.data_dir, 'qa_data.pkl')
    if os.path.isfile(qa_data):
        with open(qa_data, 'rb') as f:
            data = pickle.load(f)
            return data

    # Else, prepare data
    train_q_json = os.path.join(args.data_dir, 'MultipleChoice_mscoco_train2014_questions.json')
    train_a_json = os.path.join(args.data_dir, 'mscoco_train2014_annotations.json')
    val_q_json = os.path.join(args.data_dir, 'MultipleChoice_mscoco_val2014_questions.json')
    val_a_json = os.path.join(args.data_dir, 'mscoco_val2014_annotations.json')

    
    print('Training questions preparation')
    with open(train_q_json) as f:
        train_q = json.loads(f.read())

    print('Training answers preparation')
    with open(train_a_json) as f:
        train_a = json.loads(f.read())

    print('Validation questions preparation')
    with open(val_q_json) as f:
        val_q = json.loads(f.read())

    print('Validation answers preparation')
    with open(val_a_json) as f:
        val_a = json.loads(f.read())

    print('Amoust of Training questions: {0}'.format(len(train_q['questions'])))
    print('Amoust of Training Answers: {0}'.format(len(train_a['annotations'])))
    print('Amoust of Validation questions number: {0}'.format(len(val_q['questions'])))
    print('Amoust of Validation Answers: {0}'.format(len(val_a['annotations'])))

    # Starting making QA pairs
    questions = train_q['questions'] + val_q['questions']
    answers = train_a['annotations'] + val_a['annotations']

    answer_vocab = make_answer_vocab(answers)
    question_vocab, max_question_length = make_questions_vocab(questions, answers, answer_vocab)
    print('Max Question Length: ' + str(max_question_length))
    word_regex = re.compile(r'\w+') # Match [a-zA-Z0-9]
    train_data = []
    for i, question in enumerate(train_q['questions']):
        ans = train_a['annotations'][i]['multiple_choice_answer']
        if ans in answer_vocab:
            train_data.append({
                'image_id': train_a['annotations'][i]['image_id'],
                'question': np.zeros(max_question_length),
                'answer': answer_vocab[ans]
            })
            question_words = re.findall(word_regex, question['question'])
            base = max_question_length - len(question_words)
            for i in range(0, len(question_words)):
                train_data[-1]['question'][base + i] = question_vocab[question_words[i]]
    print('Actual Training Data num' + str(len(train_data)))
    val_data = []

    for i, question in enumerate(val_q['questions']):
        ans = val_a['annotations'][i]['multiple_choice_answer']
        if ans in answer_vocab:
            val_data.append({
                'image_id' : val_a['annotations'][i]['image_id'],
                'question' : np.zeros(max_question_length),
                'answer' : answer_vocab[ans]
            })
            question_words = re.findall(word_regex, question['question'])

            base = max_question_length - len(question_words)
            for i in range(0, len(question_words)):
                val_data[-1]['question'][base + i] = question_vocab[question_words[i]]

    print('Actual Validation Data num' + str(len(val_data)))

    data = {
        'train' : train_data,
        'val' : val_data,
        'answer_vocab' : answer_vocab,
        'question_vocab' : question_vocab,
        'max_question_length' : max_question_length
    }

    print('Saving qa data')
    with open(qa_data, 'wb') as f:
        pickle.dump(data, f)

    vocab_file = os.path.join(args.data_dir, 'vocab_file.pkl')
    with open(vocab_file, 'wb') as f:
        vocab_data = {
            'answer_vocab' : data['answer_vocab'],
            'question_vocab' : data['question_vocab'],
            'max_question_length' : data['max_question_length']
        }
        pickle.dump(vocab_data, f)
    return data

def get_question_answer_vocab(data_dir):
    vocab_file = os.path.join(data_dir, 'vocab_file.pkl')
    vocab_data = pickle.load(open(vocab_file, 'rb'))
    return vocab_data

def make_answer_vocab(answers):
    top_n = 1000
    answer_frequency = {}
    for annotation in answers:
        answer = annotation['multiple_choice_answer']
        if answer in answer_frequency:
            answer_frequency[answer] += 1
        else:
            answer_frequency[answer] = 1

    answer_frequency_tuples = [ (-frequency, answer) for answer, frequency in answer_frequency.items()]
    answer_frequency_tuples.sort()
    answer_frequency_tuples = answer_frequency_tuples[0:top_n-1]

    answer_vocab = {}
    for i, ans_freq in enumerate(answer_frequency_tuples):
        # print i, ans_freq
        ans = ans_freq[1]
        answer_vocab[ans] = i

    answer_vocab['UNK'] = top_n - 1
    return answer_vocab


def make_questions_vocab(questions, answers, answer_vocab):
    word_regex = re.compile(r'\w+')
    question_frequency = {}

    max_question_length = 0
    for i,question in enumerate(questions):
        ans = answers[i]['multiple_choice_answer']
        count = 0
        if ans in answer_vocab:
            question_words = re.findall(word_regex, question['question'])
            for qw in question_words:
                if qw in question_frequency:
                    question_frequency[qw] += 1
                else:
                    question_frequency[qw] = 1
                count += 1
        if count > max_question_length:
            max_question_length = count


    qw_freq_threhold = 0
    qw_tuples = [ (-frequency, qw) for qw, frequency in question_frequency.items()]

    qw_vocab = {}
    for i, qw_freq in enumerate(qw_tuples):
        frequency = -qw_freq[0]
        qw = qw_freq[1]
        # print frequency, qw   
        #if frequency > qw_freq_threhold:
            # +1 for accounting the zero padding for batch training
        qw_vocab[qw] = i + 1
        #else:
            #break

    qw_vocab['UNK'] = len(qw_vocab) + 1

    return qw_vocab, max_question_length

def load_image_features(directory, mode):
    #image_features = None
    #image_id_list = None
    with h5py.File(os.path.join(directory, (mode + '_fc7.h5')),'r') as hf:
        image_features = np.array(hf.get('fc7_features'))
    with h5py.File(os.path.join(directory, (mode + '_image_id_list.h5')),'r') as hf:
        image_id_list = np.array(hf.get('image_id_list'))
    return image_features, image_id_list


