import os
import string
import random
from detector import *
from flask import Flask, request, render_template, jsonify

#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz"
#modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_kpts_coco17_tpu-32.tar.gz"
#imagePath = "/home/raj/pix/ret/dog_bike_car.jpg"

modelURL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"
classFile = "coco.names"

app = Flask(__name__)

@app.route('/1/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No File Given"

        file = request.files['image']
        rand_id = ''.join(random.choices(string.ascii_lowercase + string.digits, k = 5))
        fp = '/tmp/' + rand_id + '.jpg'
        file.save(fp)
        threshold = 0.6
        detector = Detector()
        detector.readClasses(classFile)
        detector.downloadModel(modelURL)

        detector.loadModel()
        x = detector.predictImage(fp, threshold)
        res = {
            "count": x
        }
        return jsonify(res)
    
    return "Make a proper POST request"

if __name__=='__main__':
    app.run(debug=True)
