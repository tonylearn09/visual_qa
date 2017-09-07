import tensorflow as tf
import numpy as np
import argparse
import loader
import lstm
import os

def train(args):
    print('Reading QA data')
    qa_data = loader.load_qas(args)

    print('Reading image features')
    image_features, image_id_list = loader.load_image_features(args.data_dir, 'train')
    #print('FC7 features shape: .{0}'.format(image_features.shape))
    #print('image_id_list shape: {0}'.format(image_id_list.shape))
    
    #image_id_map = {}
    #for i in range(len(image_id_list)):
        #image_id_map[image_id_list[i]] = i

    image_id_map = {k: v for v, k in enumerate(image_id_list)}
    ans_map = {qa_data['answer_vocab'][ans] : ans for ans in qa_data['answer_vocab']}

    model_options = {
        'n_lstm_layers' : args.n_lstm_layers,
        'rnn_size' : args.rnn_size,
        'embedding_size' : args.embedding_size,
        'word_emb_dropout' : args.word_emb_dropout,
        'image_dropout' : args.image_dropout,
        'img_feature_dim' : args.img_feature_dim,
        'lstm_steps' : qa_data['max_question_length'] + 1,
        'q_vocab_size' : len(qa_data['question_vocab']),
        'ans_vocab_size' : len(qa_data['answer_vocab'])
    }

    model = lstm.Lstm(model_options)
    input_tensors, t_loss, t_accuracy, t_p = model.build_model()
    train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(t_loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if args.restore:
            ckpt = tf.train.latest_checkpoint(args.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print('restore from checkpoint {0}'.format(ckpt))

        for i in range(args.epochs):
            batch_no = 0

            while (batch_no * args.batch_size) < len(qa_data['train']):
                sentence, answer, fc7 = get_training_batch(batch_no, args.batch_size, 
                                                           image_features, image_id_map, qa_data, 'train')
                _, loss_value, accuracy, pred = sess.run([train_op, t_loss, t_accuracy, t_p], feed_dict={
                    input_tensors['fc7']: fc7,
                    input_tensors['sentence']: sentence,
                    input_tensors['answer']: answer
                })
                batch_no += 1
                '''for idx, p in enumerate(pred):
                    print('Predicted: {0}  Actual: {1}'.format(ans_map[p], ans_map[np.argmax(answer[idx])]))'''
                print('Epoch {0} Batch {1},  Loss: {2},  Accuracy: {3}'.format(i, batch_no, loss_value, accuracy))
                #print('---------------------------------------------------------')

            # Save every epoch
            save_path = saver.save(sess, os.path.join(args.checkpoint_dir, 'my-model')) 

        
def get_training_batch(batch_no, batch_size, image_features, image_id_map, qa_data, mode):
    #qa = None
    if mode == 'train':
        qa_mode_data = qa_data['train']
    else:
        qa_mode_data = qa_data['val']

    start = (batch_no * batch_size) % len(qa_mode_data)
    end = min(len(qa_mode_data), start + batch_size)
    num_examples = end - start
    sentence = np.ndarray((num_examples, qa_data['max_question_length']), dtype = 'int32')
    answer = np.zeros((num_examples, len(qa_data['answer_vocab'])))
    fc7 = np.ndarray((num_examples, 4096))

    #count = 0
    for i in range(start, end):
        sentence[i-start, :] = qa_mode_data[i]['question'][:]
        answer[i-start, qa_mode_data[i]['answer']] = 1.0
        fc7_index = image_id_map[qa_mode_data[i]['image_id']]
        fc7[i-start, :] = image_features[fc7_index][:]
        #count += 1

    return sentence, answer, fc7



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_lstm_layers', type=int, default=2,
                       help='numer of layers in lstm')
    parser.add_argument('--img_feature_dim', type=int, default=4096,
                       help='image feature length (dimension)')
    parser.add_argument('--rnn_size', type=int, default=512,
                       help='size of lstm internal state')
    parser.add_argument('--embedding_size', type=int, default=512,
                       help='size of word embedding'),
    parser.add_argument('--word_emb_dropout', type=float, default=0.5,
                       help='dropout for word embeddding')
    parser.add_argument('--image_dropout', type=float, default=0.5,
                       help='dropout for image embedding')
    parser.add_argument('--data_dir', type=str, default='Data',
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=20,
                       help='Batch Size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='learning rate for training')
    parser.add_argument('--epochs', type=int, default=20,
                       help='number of epochs')  
    parser.add_argument('--restore', type=bool, default=False,
                       help='Whether to restore')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                        help='path to checkpoint directory')

    args = parser.parse_args()
    train(args)
