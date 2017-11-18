import tensorflow as tf
from scipy import misc
import os
import sys
import numpy as np
import argparse
import utils
import h5py
import loader

from FCRN_DepthPrediction.tensorflow import predict
from FCRN_DepthPrediction.tensorflow import models

def get_minibatches(input_set, batchsize):
    batch_image = np.ndarray((batchsize, 224, 224, 3))
    actual = 0
    count = 0
    for idx in input_set:
        image = os.path.join(args.data_dir, '%s2014/COCO_%s2014_%.12d.jpg' % (args.mode, args.mode, idx))
        batch_image[actual, :, :, :] = utils.load_image_array(image)
        actual += 1
        count += 1
        if actual >= batchsize or count >= len(input_set):
            yield batch_image[0: actual, :, :, :], actual
            actual = 0



def img_extract(args):
    model_file = open(args.path, mode='rb')
    model = model_file.read()
    model_file.close()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(model)
    
    images = tf.placeholder('float32', [None, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={'images': images})

    graph = tf.get_default_graph()

    all_data = loader.load_qas(args)
    if args.mode == 'train':
        qa_data = all_data['train']
    else:
        qa_data = all_data['val']

    unique_image_id = set()
    #image_ids = {}
    for qa in qa_data:
        #image_ids[qa['image_id']] = 1
        unique_image_id.add(qa['image_id'])

    #image_id_list = [img_id for img_id in image_ids]
    #print('Total images' + str(len(image_id_list)))
    print('Total images' + str(len(unique_image_id)))



    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    input_node = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    net = models.ResNet50UpProj({'data': input_node}, args.batch_size, 1, False)
    with tf.Session(config=sess_config) as sess:
        depth_features = np.ndarray((len(unique_image_id), 112*112))
        print('Loading the depth prediction model')
        # Use to load from ckpt file
        saver = tf.train.Saver()     
        saver.restore(sess, 'FCRN_DepthPrediction/tensorflow/NYU_FCRN.ckpt')

        count = 0
        temp = get_minibatches(unique_image_id, args.batch_size)
        for j, k in temp:
            batch_image = j
            actual_batch_size = k

            print('Start batch {0}'.format(count))
            # depth
            depth_feature = sess.run(net.get_output(), feed_dict={input_node: batch_image})
            depth_feature = depth_feature.reshape(actual_batch_size, -1)
            depth_features[count*args.batch_size: count*args.batch_size+actual_batch_size] = depth_feature
            count += 1

        print('Saving depth features')
        h5f_depth = h5py.File(os.path.join(args.data_dir, args.mode + '_depth.h5'), 'w')
        h5f_depth.create_dataset('depth_features', data=depth_features)
        h5f_depth.close()

    with tf.Session(config=sess_config) as sess:
        print('Start extracting image features')
        #features = np.ndarray((len(image_id_list), 4096))
        features = np.ndarray((len(unique_image_id), 4096))

        count = 0
        temp = get_minibatches(unique_image_id, args.batch_size)
        for j, k in temp:
            batch_image = j
            actual_batch_size = k

            print('Start batch {0}'.format(count))
            # fc7
            fc7_tensor = graph.get_tensor_by_name('import/Relu_1:0')
            fc7_feature = sess.run(fc7_tensor, feed_dict={images: batch_image})
            features[count*args.batch_size: count*args.batch_size+actual_batch_size] = fc7_feature
            count += 1

        print('Saving fc7 features')
        h5f_fc7 = h5py.File(os.path.join(args.data_dir, args.mode + '_fc7.h5'), 'w')
        h5f_fc7.create_dataset('fc7_features', data=features)
        h5f_fc7.close()


        print('Saving image id list')
        h5f_image_id_list = h5py.File(os.path.join(args.data_dir, args.mode + '_image_id_list.h5'), 'w')
        h5f_image_id_list.create_dataset('image_id_list', data=list(unique_image_id))
        h5f_image_id_list.close()


    # Extract depth info





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str, help='which part of data to extract img feature (train / val)')
    parser.add_argument('--path', type=str, default='Model/vgg16.tfmodel',
                        help='Model to extract feature')
    parser.add_argument('--data_dir', type=str, default='Data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    args = parser.parse_args()
    img_extract(args)
