from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from skimage import measure
import argparse
import imutils
import cv2
import time
import random

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model",
	help="path to pre-trained smile detector CNN")
args = vars(ap.parse_args())

#this variable defines which CNN model is going to be use for inference in the
#found rois

model = load_model('epoch_30_mobilenetv2.hdf5')

#classes of objects we are trying to detect

CLASSES = ["botellas_plastico", "latas_aluminio", "papel_carton"]

#colors use for annotating the objects found, three colors since we have three objects

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

#defining the video capture method to extract the frames from the camera

ls = cv2.VideoCapture(2)
time.sleep(2.0)

#defining the mixture of gaussians background subtractor

bgs = cv2.createBackgroundSubtractorMOG2(history = 1500, detectShadows = False)

#font use for annotating the objects and info displayed

font = cv2.FONT_HERSHEY_SIMPLEX

#variables for calculating FPS speed of inference and BGS operation

prev_frame_time = 0

new_frame_time = 0

prev_frame_time_s = 0

new_frame_time_s = 0

#kernel use for preprocessing in morphological operations

kernel = np.ones((11,11),np.uint8)

#function for producing a random color from {red, green, blue}
#This is use in the img_correction function for increasing performance in saliency operation

def random_color():
	rgbl = [255, 0, 0]
	random.shuffle(rgbl)
	return tuple(rgbl)

#function for making roi square for inference given that the roi produce is not square
#and the CNN requires an square image

def coord(x_min, x_max, y_min, y_max):

	dX = x_max - x_min
	dY = y_max - y_min

	spacer_x = int(dX/2)
	spacer_y = int(dY/2)

	if dX > dY:
		y_min_dummy = y_min
		y_max_dummy = y_max
		y_min = (spacer_y + y_min_dummy) - spacer_x
		y_max = (spacer_y + y_min_dummy) + spacer_x

		if y_min < 0:

			y_min_dummy = y_min
			y_max_dummy = y_max

			y_min = 0

			y_max = y_max_dummy + (-1 * y_min_dummy)

		if y_max > height:

			y_min_dummy = y_min
			y_max_dummy = y_max

			y_min = y_min_dummy - (y_max_dummy - height)

			y_max = height

	else:
		x_min_dummy = x_min
		x_max_dummy = x_max

		x_min = (spacer_x + x_min_dummy) - spacer_y
		x_max = (spacer_x + x_min_dummy) + spacer_y

		if x_min < 0:

			x_min_dummy = x_min
			x_max_dummy = x_max

			x_min = 0

			x_max = x_max_dummy + (-1 * x_min_dummy)

		if x_max > width:

			x_min_dummy = x_min
			x_max_dummy = x_max

			x_min = x_min_dummy - (x_max_dummy - width)

			x_max = width

	return x_min, x_max, y_min, y_max

#function for preprocessing the image prior to running saliency operation on it
#it grabs the contours and draw random color points on the countors
#this helps the saliency operation, specially for transparent objects (plastic bottles)

def img_correction(image):

	#transform image to gray scale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#thresholding the image, simple thresholding NOT Otsu
	ret, thresh = cv2.threshold(gray, 250, 255, 0)
	#finding the countors on the thresh map, not do linear aproximation
	#in order for it to be points
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	#making copy of image to not affect frame to be displayed
	dummy_img = image.copy()
	#drawing the countuors found with random countors
	for i, cont in enumerate(contours):

		cv2.drawContours(dummy_img, contours, i, random_color(), 6)

	return dummy_img

#function for perfoming the saliency operation to extract the objects that resalt in the image

def saliency(image, saliency_mask = None):

	#defining the saliency operation, spectral residual
	saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
	#saliency = cv2.saliency.StaticSaliencyFineGrained_create()
	#computing the saliency maps
	(success, saliencyMap) = saliency.computeSaliency(image)
	#changing the to 8-bit data type
	saliencyMap = (saliencyMap * 255).astype("uint8")
	#perform difference just if saliency_mask is provided
	if saliency_mask is not None:
		saliencyMap = cv2.absdiff(saliencyMap, saliency_mask)
	#performing thresholding OTSU thresholding in the saliencyMap
	saliencyMap = cv2.GaussianBlur(saliencyMap.astype("uint8"), (11,11), 0)
	threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	return saliencyMap, threshMap

def saliency_subs(image, saliency_mask):

	#defining the saliency operation, spectral residual
	saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
	#saliency = cv2.saliency.StaticSaliencyFineGrained_create()
	#computing the saliency maps
	(success, saliencyMap) = saliency.computeSaliency(image)
	#changing the to 8-bit data type
	saliencyMap = (saliencyMap * 255).astype("uint8")
	saliencyMap = cv2.absdiff(saliencyMap, saliency_mask)
	#performing thresholding OTSU thresholding in the saliencyMap
	threshMap = cv2.threshold(saliencyMap.astype("uint8"), 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	return saliencyMap, threshMap

saliency_img = []

width_cut = 60

for i in range(1000):

	ret, frame = ls.read()

	new_frame_time = time.time()

	height, width, channels = frame.shape

	frame = frame[0:height,width_cut:width - width_cut]

	frame = imutils.resize(frame, width = 200)

	img_corr = img_correction(frame)

	saliencyMap, threshMap = saliency(frame)

	saliency_img.append(saliencyMap)

	cv2.imshow('Saliency map', imutils.resize(saliencyMap, width = 500))

	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

mean_saliency_img = np.mean(saliency_img, axis = 0)
mean_saliency_img = mean_saliency_img.astype(np.uint8)

cv2.imshow('Saliency mean', imutils.resize(mean_saliency_img, width = 500))
cv2.waitKey(0)

max_int = 0

const = 1

while True:

	ret, frame = ls.read()

	new_frame_time = time.time()

	height, width, channels = frame.shape

	frame = frame[0:height,width_cut:width - width_cut]

	height, width, channels = frame.shape

	frame_inf = frame.copy()

	frame_sal = frame.copy()

	frame_show = frame.copy()

	frame = imutils.resize(frame, width = 200)

	img_corr = img_correction(frame)

	saliencyMap, threshMap = saliency(frame)

	saliencyMap_sub, threshMap_sub = saliency(frame, mean_saliency_img)

	if np.mean(saliencyMap_sub) > 4 and np.std(saliencyMap_sub) > 14 and np.max(saliencyMap_sub) >  160:

		cv2.putText(frame_show, '[OBJECT DETECTED]', (250, 460), font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

		erosion_thresh = cv2.morphologyEx(threshMap, cv2.MORPH_CLOSE, kernel)

		erosion_thresh_sub = cv2.morphologyEx(threshMap_sub, cv2.MORPH_CLOSE, kernel)

		#thresholded_saliency = cv2.threshold(saliencyMap, 40, 255, cv2.THRESH_TOZERO)[1]
		#thresholded_saliency = cv2.GaussianBlur(thresholded_saliency, (7,7), 0)
		#thresholded_thresh = cv2.threshold(thresholded_saliency, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

		#cv2.imshow('Thresholded saliency', thresholded_saliency)
		#cv2.imshow('Otsu saliency', thresholded_thresh)

		frame_horizontal_erosion = np.concatenate((imutils.resize(erosion_thresh, width = 500), imutils.resize(erosion_thresh_sub, width = 500)), axis = 1)

		cv2.imshow('Eroded thresh map', frame_horizontal_erosion)

		img_corr = img_correction(frame_sal)

		saliencyMap_corr, threshMap_corr = saliency(img_corr)

		threshMap_corr = cv2.erode(threshMap_corr, kernel, iterations = 2)

		if np.mean(erosion_thresh) < np.mean(erosion_thresh_sub):

			summed = imutils.resize(erosion_thresh, width = width)
			summed = cv2.resize(erosion_thresh, (520, 480))

		else:

			summed = imutils.resize(erosion_thresh_sub, width = width)
			summed = cv2.resize(erosion_thresh_sub, (520, 480))

		print(summed.shape)

		print(threshMap_corr.shape)

		threshMap_summed = cv2.add(threshMap_corr, summed)

		threshMap_summed = cv2.threshold(threshMap_summed, 250, 255, cv2.THRESH_BINARY)[1]

		cnts = cv2.findContours(threshMap_summed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		cnts = imutils.grab_contours(cnts)

		for c in cnts:

			if cv2.contourArea(c) < 2000:
				print(cv2.contourArea(c))
				continue

			(x, y, w, h) = cv2.boundingRect(c)

			extension = 5

			x = x - extension if x - extension > 0 else 0
			w = w + 2*extension if w + 2*extension < width else width
			y = y - extension if y - extension > 0 else 0
			h = h + 2*extension if h + 2*extension < height else height

			x_min, x_max, y_min, y_max = coord(x, x + w, y, y + h)

			roi = frame_inf[y_min:y_max,x_min:x_max]

			try:

				roi_2 = imutils.resize(roi.copy(), width = 224)

				roi_2 = cv2.resize(roi_2, (224, 224))

				if roi_2.shape[1] != 224 or roi_2.shape[0] != 224:
					continue

				roi_cnn = preprocess_input(roi_2)

				roi_cnn = np.expand_dims(roi_cnn, axis = 0)

				preds = model.predict(roi_cnn)

				ID = np.argmax(preds, axis = 1)[0]

				prob = round(np.max(preds), 2)

				label = CLASSES[ID]

				color = COLORS[ID]

				print('[OBJECT DETECTED]:  ' + str(label))

				cv2.imshow('ROI', roi_2)

				cv2.rectangle(frame_show, (x_min, y_min), (x_max, y_max), color, 2)
				cv2.putText(frame_show, label, (x_min - 3,y_max + 20), font, 0.8, color, 1, cv2.LINE_AA)
				cv2.putText(frame_show, str(prob), (x_min - 3,y_max + 40), font, 0.8, color, 2, cv2.LINE_AA)

			except:

				print('[INFO]: frame error')

		cv2.imshow('Corrected image', img_corr)
		frame_horizontal_saliency_corr = np.concatenate((imutils.resize(saliencyMap_corr, width = 500), imutils.resize(threshMap_corr, width = 500)), axis = 1)
		cv2.imshow('Corrected maps', frame_horizontal_saliency_corr)
		cv2.imshow('Summed', imutils.resize(threshMap_summed, width = 500))

	fps = 1/(new_frame_time - prev_frame_time)

	prev_frame_time = new_frame_time

	cv2.putText(frame_show, 'fps:  ' + str(int(fps)), (10, 460), font, 0.8, (255, 255, 0), 1, cv2.LINE_AA)

	cv2.imshow('Original frame', imutils.resize(frame_show, width = 500))

	frame_horizontal_saliency = np.concatenate((imutils.resize(saliencyMap, width = 500), imutils.resize(saliencyMap_sub, width = 500)), axis = 1)
	frame_horizontal_thresh = np.concatenate((imutils.resize(threshMap, width = 500), imutils.resize(threshMap_sub, width = 500)), axis = 1)
	frame_vertical = np.concatenate((frame_horizontal_saliency, frame_horizontal_thresh), axis = 0)

	cv2.putText(frame_vertical, 'max intensity:  ' + str(int(np.max(saliencyMap))), (10, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

	max_int = np.max(saliencyMap_sub) if np.max(saliencyMap_sub) > max_int else max_int

	cv2.putText(frame_vertical, 'mean:  ' + str(np.mean(saliencyMap_sub)), (510, 390), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
	cv2.putText(frame_vertical, 'std:  ' + str(np.std(saliencyMap_sub)), (510, 410), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
	cv2.putText(frame_vertical, 'intensity:  ' + str(int(np.max(saliencyMap_sub))), (510, 430), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
	cv2.putText(frame_vertical, 'max intensity:  ' + str(int(max_int)), (510, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

	cv2.imshow('Maps', frame_vertical)



	if cv2.waitKey(1) & 0xFF == ord("q"):
		break





ls.release()
cv2.destroyAllWindows()
