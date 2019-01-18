import numpy as np
import time

#todo clip for both network and boxutility

class BBoxUtility:
	def __init__(self, image_shape, class_num = 80, anchor_cfg = {"scales": [0.5, 1], "ratios": [0.5, 1, 2]}):
		self.image_shape = image_shape
		self.anchors = []
		for level in range(4, 9):
			anchors = self.generate_anchors(image_shape[0], image_shape[1], level, scales = anchor_cfg["scales"], ratios = anchor_cfg["ratios"])
			self.anchors.append(anchors)
		self.anchors = np.concatenate(self.anchors)
		self.class_num = class_num
		self.num_anchors_per_loc = len(anchor_cfg["scales"])*len(anchor_cfg["ratios"])

	def generate_anchors(self, width, height, level, scales = [0.5, 1], ratios = [0.5, 1, 2]):
		'''
		The same logic for below code, but faster implement
		for l in range(level):
			width = (width + 1) //2
			height = (height + 1) //2
		anchors = []
		for i in range(width):
			for j in range(height):
				for scale in scales:
					for ratio in ratios:
						anchors.append([i/width, j/height, scale/np.sqrt(ratio)/width, scale*np.sqrt(ratio)/height])
		return np.array(anchors)
		'''
		for l in range(level):
			width = (width + 1) //2
			height = (height + 1) //2

		x = (np.arange(width) + 0.5) / width
		y = (np.arange(height) + 0.5) / height
		s = np.array(scales)
		r = np.array(ratios)
		x, y, s, r = np.meshgrid(x, y, s, r)
		x = x.transpose([1, 0, 2, 3]).reshape([-1])
		y = y.transpose([1, 0, 2, 3]).reshape([-1])
		s = s.transpose([1, 0, 2, 3]).reshape([-1])
		r = r.transpose([1, 0, 2, 3]).reshape([-1])
		h = s * np.sqrt(r) / height
		w = s / np.sqrt(r) / width
		anchors = np.stack([x, y, w, h], axis = -1).reshape([-1, 4])
		return anchors

	def calc_iou(self, gt, anchors):
		'''
		returns:
			ious: [gt_num, anchors_num]
		'''
		def anchors_to_boxes(anchors):
			ltx = anchors[:, 0] - anchors[:, 2]/2
			lty = anchors[:, 1] - anchors[:, 3]/2
			rbx = anchors[:, 0] + anchors[:, 2]/2
			rby = anchors[:, 1] + anchors[:, 3]/2
			return np.stack([ltx, lty, rbx, rby], axis = 1)

		gtboxes = anchors_to_boxes(gt)
		anchorboxes = anchors_to_boxes(anchors)
		ious = []
		anchorboxes_squares = (anchorboxes[:, 3] - anchorboxes[:, 1])*(anchorboxes[:, 2] - anchorboxes[:, 0])
		for gtbox in gtboxes:
			ltx = np.maximum(gtbox[0], anchorboxes[:, 0])
			lty = np.maximum(gtbox[1], anchorboxes[:, 1])
			rbx = np.minimum(gtbox[2], anchorboxes[:, 2])
			rby = np.minimum(gtbox[3], anchorboxes[:, 3])

			iwidth = np.maximum(rbx - ltx, 0)
			iheight = np.maximum(rby - lty, 0)
			intersection_squeare = iwidth*iheight
			gtbox_square = (gtbox[3]-gtbox[1])*(gtbox[2] - gtbox[0])
			iou =  intersection_squeare/(np.maximum(gtbox_square + anchorboxes_squares - intersection_squeare, 0) + 1e-10)
			ious.append(iou)
		ious = np.array(ious)

		return ious	

	def gtbox_assign(self, gt, gtcls, positive_threshold = 0.5, negative_threshold = 0.4):
		ious = self.calc_iou(gt, self.anchors)
		pos_indices = ious >= positive_threshold
		bg_indices = ious < negative_threshold
		box_label = np.zeros_like(self.anchors)
		class_label = np.zeros([self.anchors.shape[0], self.class_num+1])
		for i in range(ious.shape[0]):
			class_label[pos_indices[i, :], gtcls[i]] = 1
			box_label[pos_indices[i, :]] = gt[i, :] - self.anchors[pos_indices[i, :]]

		pos_indices = np.sum(pos_indices, axis = 0) > 0
		bg_indices = np.sum(~bg_indices, axis = 0) == 0
		class_label[bg_indices, -1] = 1
		return box_label, class_label, pos_indices, bg_indices

if __name__ == "__main__":
	boxUtil = BBoxUtility((256, 256))
	gt = np.array([[0.28888666577,0.5, 0.81,0.748484848484844], [0.35555555,0.255555555,0.5,0.32222222222222222222222], [0.2,0.5, 0.1,0.3], [0.2,0.2,0.5,0.3], [0.2,0.5, 0.1,0.3], [0.2,0.2,0.5,0.3]])
	gtcls = np.array([2, 0, 2, 0, 2, 0]).astype(int)
	aa, bb, cc, dd = boxUtil.gtbox_assign(gt, gtcls, 6)
	print(aa.shape)
	print(np.unique(aa, axis = 0))
	print(np.unique(cc))
	print(dd)

	# anchors = boxUtil.generate_anchors(256, 256, 7)
	# print(anchors)
	# gt = np.array([[0.28888666577,0.5, 0.1,0.748484848484844]])
	# gtcls = [2]
	# ious = boxUtil.calc_iou(gt, anchors)
	# print(np.max(ious))
	# print(anchors[np.argmax(ious)])	

	# def calc_iou(self, gt, anchors):
	# 	def anchors_to_boxes(anchors):
	# 		ltx = anchors[:, 0] - anchors[:, 2]/2
	# 		lty = anchors[:, 1] - anchors[:, 3]/2
	# 		rbx = anchors[:, 0] + anchors[:, 2]/2
	# 		rby = anchors[:, 1] + anchors[:, 3]/2
	# 		return np.stack([ltx, lty, rbx, rby], axis = 1)
	# 	def boxes_square(boxes):
	# 		return (boxes[:, :,3] - boxes[:, :, 1])*(boxes[:, :, 2] - boxes[:, :, 0])
	# 	gtboxes = anchors_to_boxes(gt)
	# 	anchorboxes = anchors_to_boxes(anchors)
	# 	gtboxes = np.expand_dims(gtboxes, 1)
	# 	gtboxes_square = boxes_square(gtboxes)
	# 	anchorboxes = np.expand_dims(anchorboxes, 0)
	# 	anchorboxes_squares = boxes_square(anchorboxes)
	# 	gtboxes = np.tile(gtboxes, (anchorboxes.shape[1], 1))
	# 	anchorboxes = np.tile(anchorboxes, (gtboxes.shape[0], 1, 1))
	# 	ltx = np.maximum(gtboxes[:,:,0], anchorboxes[:,:, 0])
	# 	lty = np.maximum(gtboxes[:,:,1], anchorboxes[:,:, 1])
	# 	rbx = np.minimum(gtboxes[:,:,2], anchorboxes[:,:, 2])
	# 	rby = np.minimum(gtboxes[:,:,3], anchorboxes[:,:, 3])
	# 	iwidth = np.maximum(rbx - ltx, 0)
	# 	iheight = np.maximum(rby - lty, 0)
	# 	intersection_squeare = iwidth*iheight
	# 	ious = intersection_squeare / (np.maximum(0, gtboxes_square + anchorboxes_squares - intersection_squeare) + 1e-10)
		