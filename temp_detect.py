MODEL = 'ssd_mobilenet_v2_coco_2018_03_29' # BOXES
import glob
import os
import sys
import six.moves.urllib as urllib
import tarfile
import tensorflow.compat.v1 as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

TF_MODELS_RESEARCH_PATH = '/home/raj/dox/ompet/models/research'
sys.path.append(TF_MODELS_RESEARCH_PATH)

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import numpy as np
from PIL import Image


MODEL_FILE = MODEL + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_FROZEN_INFERENCE_GRAPH = os.path.join(MODEL, 'frozen_inference_graph.pb')
PATH_TO_GRAPH_LABELS = os.path.join(TF_MODELS_RESEARCH_PATH, 'object_detection', 'data', 'mscoco_label_map.pbtxt')
MAX_NUM_CLASSES = 90

TEST_IMAGE_MASK = sys.argv[1]  # python detect.py /tmp/myfile.jpg


if not os.path.exists(PATH_TO_FROZEN_INFERENCE_GRAPH):
    if not os.path.exists(MODEL_FILE):
        print("Download model file from: " + DOWNLOAD_BASE + MODEL_FILE)
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    print("Extract frozen inference graph to: " + PATH_TO_FROZEN_INFERENCE_GRAPH)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_INFERENCE_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_GRAPH_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=MAX_NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session(config=config) as sess:
            # Get handles to input and output tensors
            ops = 
            all_tensor_names = {output.name for op in tf.get_default_graph().get_operations() for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)

            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])

            if 'detection_masks' in tensor_dict:
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]

            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
            else:
                output_dict['detection_masks'] = None

            return output_dict


def ObjectNumber(path):
    image = Image.open(path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    num_detections = int(output_dict.pop('num_detections'))
    return num_detections

print(ObjectNumber(sys.argv[1]))
