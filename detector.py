import cv2, time,os, tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(20)


class Detector:
	def __init__(self):
		pass

	def readClasses(self, classesFilePath):
		with open(classesFilePath, 'r') as f:
			self.classesList = f.read().splitlines()


		self.colorList = np.random.uniform(low = 0 , high = 255, size=(len(self.classesList), 3))
		print(len(self.classesList), len(self.colorList))

	def downloadModel(self, modelURL):
		fileName = os.path.basename(modelURL)
		self.modelName = fileName[:fileName.index('.')]

		self.cacheDir = "./pretained_models"

		os.makedirs(self.cacheDir, exist_ok = True)
		get_file(fname=fileName, origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)
	def loadModel(self):
		print("Loading Model" + self.modelName)
		tf.keras.backend.clear_session()
		self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
		print("Model " + self.modelName + " loaded successfully...")

	def createBoundingBox(self,image, threshold = 0.5):
		inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
		inputTensor = tf.convert_to_tensor(inputTensor, dtype = tf.uint8)
		inputTensor = inputTensor[tf.newaxis,...]

		detections = self.model(inputTensor)

		bboxs = detections['detection_boxes'][0].numpy()
		classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
		classScores = detections['detection_scores'][0].numpy()

		num_detections = int(detections.pop('num_detections'))

		imH, imW, imC = image.shape

		bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)

		'''
		for i in bboxIdx:
			bbox = tuple(bboxs[i].tolist())
			classConfidence = round(100*classScores[i])
			classIndex = classIndexes[i]

			classLabelText = self.classesList[classIndex]
			classColor = self.colorList[classIndex]
			displayText = '{}: {}%'.format(classLabelText, classConfidence)

			ymin, xmin, ymax, xmax = bbox

			xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH)
			xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)

			cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1 )
			cv2.putText(image, displayText, (xmin, ymin - 10 ), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2  )

			lineWidth = min(int((xmax - xmin ) * 0.2 ), int((ymax - ymin ) * 0.2))
		'''

		return bboxIdx.shape[0]

	def predictImage(self,imagePath, threshold):
		image = cv2.imread(imagePath)

		bboxImage =  self.createBoundingBox(image, threshold)
		return bboxImage
