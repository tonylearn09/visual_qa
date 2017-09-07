import tensorflow as tf
import numpy as np
import argparse
import loader
import lstm
import re
import utils
import os

def predict():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default = 'Data/cat.jpeg',
                       help='Image Path')
    #parser.add_argument('--model_path', type=str, default = 'Data/Models/model2.ckpt',
                       #help='Model Path')
    parser.add_argument('--n_lstm_layers', type=int, default=2,
                       help='num_lstm_layers')
    parser.add_argument('--img_feature_dim', type=int, default=4096,
                       help='fc7_feature_length')
    parser.add_argument('--rnn_size', type=int, default=512,
                       help='rnn_size')
    parser.add_argument('--embedding_size', type=int, default=512,
                       help='embedding_size'),
    parser.add_argument('--word_emb_dropout', type=float, default=0.5,
                       help='word_emb_dropout')
    parser.add_argument('--image_dropout', type=float, default=0.5,
                       help='image_dropout')
    parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
    parser.add_argument('--model_dir', type=str, default='Model',
                       help='model directory')
    parser.add_argument('--question', type=str, default='Which animal is this?',
                       help='Question')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                        help='path to checkpoint directory')
    args = parser.parse_args()

    print('Image: '+ args.image_path)
    print('Question: ' + args.question)

    vocab_data = loader.get_question_answer_vocab(args.data_dir)
    qvocab = vocab_data['question_vocab']
    q_map = { vocab_data['question_vocab'][qw] : qw for qw in vocab_data['question_vocab']}

    fc7_features = utils.extract_fc7_features(args.image_path, os.path.join(args.model_dir, 'vgg16.tfmodel'))

    model_options = {
        'n_lstm_layers' : args.n_lstm_layers,
        'rnn_size' : args.rnn_size,
        'embedding_size' : args.embedding_size,
        'word_emb_dropout' : args.word_emb_dropout,
        'image_dropout' : args.image_dropout,
        'img_feature_dim' : args.img_feature_dim,
        'lstm_steps' : vocab_data['max_question_length'] + 1,
        'q_vocab_size' : len(vocab_data['question_vocab']),
        'ans_vocab_size' : len(vocab_data['answer_vocab'])
    }

    question_vocab = vocab_data['question_vocab']
    word_regex = re.compile(r'\w+')
    question_ids = np.zeros((1, vocab_data['max_question_length']), dtype = 'int32')
    question_words = re.findall(word_regex, args.question)
    base = vocab_data['max_question_length'] - len(question_words)
    for i in range(0, len(question_words)):
        if question_words[i] in question_vocab:
            question_ids[0][base + i] = question_vocab[ question_words[i] ]
        else:
            question_ids[0][base + i] = question_vocab['UNK']

    ans_map = { vocab_data['answer_vocab'][ans] : ans for ans in vocab_data['answer_vocab']}
    model = lstm.Lstm(model_options)
    input_tensors, t_prediction, t_ans_probab = model.build_generator()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(args.checkpoint_dir)
    if ckpt:
        saver.restore(sess, ckpt)
        print('restore from checkpoint {0}'.format(ckpt))
    #saver.restore(sess, args.model_path)

    pred, answer_probab = sess.run([t_prediction, t_ans_probab], feed_dict={
        input_tensors['fc7']:fc7_features,
        input_tensors['sentence']:question_ids,
    })


    print('Ans: {0}'.format(ans_map[pred[0]]))
    answer_probab_tuples = [(-answer_probab[0][idx], idx) for idx in range(len(answer_probab[0]))]
    answer_probab_tuples.sort()
    print('Top Answers')
    for i in range(5):
        print(ans_map[ answer_probab_tuples[i][1] ])


if __name__ == '__main__':
    predict()
