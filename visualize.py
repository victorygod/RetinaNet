import numpy as np
import cv2

def visualize(img, boxes, scores):
	for i, box in enumerate(boxes):
		if scores[i]>0.1:
			xmin, ymin, xmax, ymax = box * 256
			cv2.rectangle(img, (int(ymin), int(xmin)), (int(ymax), int(xmax)), (255,255,0), 2)
	cv2.imwrite("output.jpg", img)

def visualize_label(img, boxes):
	img = img.copy()
	for box in boxes:
		x, y, w, h = box * 256
		xmin = x - w/2
		ymin = y - h/2
		xmax = x+w/2
		ymax = y+h/2
		cv2.rectangle(img, (int(ymin), int(xmin)), (int(ymax), int(xmax)), (255,0,0), 2)
	cv2.imwrite("label.jpg", img)