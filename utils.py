import numpy as np
import tensorflow as tf
from scipy import misc

def load_image_array(image_file):
    img = misc.imread(image_file)
    # For grayscale image
    if len(img.shape) == 2:
        img_new = np.ndarray( (img.shape[0], img.shape[1], 3), dtype = 'float32')   
        img_new[:,:,0] = img
        img_new[:,:,1] = img
        img_new[:,:,2] = img
        img = img_new

    img_resized = misc.imresize(img, (224, 224, 3))
    # For vgg, rgb should be 0~1
    return (img_resized/255.0).astype('float32')

# FOR PREDICTION ON A SINGLE IMAGE
def extract_fc7_features(image_path, model_path):
    vgg_file = open(model_path, 'rb')
    vgg16raw = vgg_file.read()
    vgg_file.close()

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(vgg16raw)
    images = tf.placeholder("float32", [None, 224, 224, 3])
    tf.import_graph_def(graph_def, input_map={ "images": images })
    graph = tf.get_default_graph()

    with tf.Session() as sess:
        image_array = load_image_array(image_path)
        fc7_tensor = graph.get_tensor_by_name("import/Relu_1:0")
        fc7_features = sess.run(fc7_tensor, feed_dict = {images: image_feed[np.newaxis, :]})
        sess.close()
    return fc7_features
