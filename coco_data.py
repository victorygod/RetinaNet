from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from scipy import misc
import numpy as np
from bbox_util import BBoxUtility

filePath = "../COCO/annotations/instances_train2014.json"
coco = COCO(filePath)

imgIds = coco.getImgIds()
imgs = coco.loadImgs(imgIds)

catIds = coco.getCatIds()
catDict = {}
for i in range(len(catIds)):
	catDict[catIds[i]] = i

class DataLoader:
	def __init__(self, bboxUtil):
		filePath = "../COCO/annotations/instances_train2014.json"
		self.coco = COCO(filePath)
		imgIds = self.coco.getImgIds()
		self.imgs_info = coco.loadImgs(imgIds)
		np.random.shuffle(self.imgs_info)
		self.img_dir = "../COCO/train2014/"
		self.cursor = 0
		self.epoch = 0
		self.bboxUtil = bboxUtil
		self.img_size = bboxUtil.img_shape


	def next_batch(self, batch_size = 20):
		imgs = []
		box_labels = []
		class_labels = []
		pos_indices = []
		bg_indices = []
		for i in range(batch_size):
			img_info = self.imgs_info[self.cursor]
			self.cursor+=1
			img = misc.imread(self.img_dir+img_info["file_name"])
			img = misc.imresize(img, self.img_size)
			imgs.append(img)

			bbox = []
			bbid = []
			annIds = self.coco.getAnnIds(imgIds = img_info["id"])
			anns = self.coco.loadAnns(annIds)
			for ann in anns:
				bbid.append(catDict[ann["category_id"]])
				y, x, h, w = ann["bbox"]
				bbox.append([x/self.img_size[0], y/self.img_size[1], w/self.img_size[0], h/self.img_size[1]])
			
			box_label, class_label, pos_index, bg_index = self.bboxUtil.gtbox_assign(np.array(bbox), np.array(bbid).astype(int))
			
			box_labels.append(box_label)
			class_labels.append(class_label)
			pos_indices.append(pos_index)
			bg_indices.append(bg_index)

			if self.cursor>=len(self.imgs_info):
				self.cursor = 0
				np.random.shuffle(self.imgs_info)
				self.epoch+=1

		pos_indices = np.array(pos_indices)
		pos_indices = np.expand_dims(pos_indices, axis = -1)
		bg_indices = np.array(bg_indices)
		bg_indices = np.expand_dims(bg_indices, axis = -1)

		return imgs, np.array(box_labels), np.array(class_labels), pos_indices, bg_indices

if __name__ == "__main__":
	dataloader = DataLoader()
	imgs, bl, cl, pi, bi = dataloader.next_batch()
	print(np.shape(imgs))
	print(bl.shape)
	print(np.unique(bl))
	print(cl.shape)
	print(np.unique(cl))
	print(pi.shape)
	print(np.unique(pi))
	print(bi.shape)
