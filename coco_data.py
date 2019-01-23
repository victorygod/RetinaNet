from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from scipy import misc
import numpy as np
from bbox_util import BBoxUtility
import visualize

filePath = "../COCO/annotations/instances_train2014.json"
coco = COCO(filePath)


catIds = coco.getCatIds()
catDict = {}
for i in range(len(catIds)):
	catDict[catIds[i]] = i

class DataLoader:
	def __init__(self, bboxUtil, mode = "train"):
		filePath = "../COCO/annotations/instances_" + mode + "2014.json"
		self.coco = COCO(filePath)
		imgIds = self.coco.getImgIds()
		self.imgs_info = self.coco.loadImgs(imgIds)[:10]
		np.random.shuffle(self.imgs_info)
		self.img_dir = "../COCO/" + mode + "2014/"
		self.cursor = 0
		self.epoch = 0
		self.bboxUtil = bboxUtil
		self.img_size = bboxUtil.image_shape


	def next_batch(self, batch_size = 20):
		imgs = []
		box_labels = []
		class_labels = []
		pos_indices = []
		bg_indices = []
		for i in range(batch_size):
			bbox = []
			bbid = []
			img_info = self.imgs_info[self.cursor]
			annIds = self.coco.getAnnIds(imgIds = img_info["id"])
			while len(annIds) == 0:
				self.cursor+=1
				if self.cursor>=len(self.imgs_info):
					self.cursor = 0
					np.random.shuffle(self.imgs_info)
					self.epoch+=1
				img_info = self.imgs_info[self.cursor]
				annIds = self.coco.getAnnIds(imgIds = img_info["id"])
			
			img = misc.imread(self.img_dir+img_info["file_name"])
			if len(img.shape)<3:
				img = np.stack([img, img, img], axis = -1)
			ori_img_shape = img.shape
			img = misc.imresize(img, self.img_size, mode = "RGB")
			imgs.append(img)

			self.cursor+=1
			anns = self.coco.loadAnns(annIds)
			for ann in anns:
				bbid.append(catDict[ann["category_id"]])
				y, x, h, w = ann["bbox"]
				x+=w/2
				y+=h/2
				bbox.append([x/ori_img_shape[0], y/ori_img_shape[1], w/ori_img_shape[0], h/ori_img_shape[1]])

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
	boxUtil = BBoxUtility((256, 256), 80)
	dataloader = DataLoader(boxUtil, mode = "train")
	imgs, bl, cl, pi, bi = dataloader.next_batch()
	# print(np.shape(imgs))
	# print(bl.shape)
	# print(np.unique(bl))
	# print(cl.shape)
	# print(np.unique(cl))
	# print(pi.shape)
	# print(np.unique(pi))
	# print(bi.shape)

