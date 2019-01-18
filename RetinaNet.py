import tensorflow as tf
import numpy as np
from coco_data import DataLoader
from bbox_util import BBoxUtility
from loss import focal_loss, reg_loss

activation = tf.nn.relu

class RetinaNet:
	def __init__(self, class_num = 80):
		self.class_num = class_num
		self.input_tensor = tf.placeholder(tf.float32, (None, None, None, 3))
		self.global_step = tf.train.create_global_step()

	def build(self, istraining = True, num_anchors_per_loc = 6):
		batch_size = tf.shape(self.input_tensor)[0]
		features = self.build_FPN(self.input_tensor, istraining = istraining)
		class_pred = []
		box_pred = []
		for level, C in enumerate(features):
			class_outputs = self._class_subnet(C, num_anchors_per_loc, level, istraining = istraining)
			box_outputs = self._box_subnet(C, num_anchors_per_loc, level, istraining = istraining)
			class_outputs = tf.reshape(class_outputs, shape = [batch_size, -1, self.class_num])
			box_outputs = tf.reshape(box_outputs, shape = [batch_size, -1, 4])
			class_pred.append(class_outputs)
			box_pred.append(box_outputs)
		return tf.concat(box_pred, axis = 1), tf.concat(class_pred, axis = 1)

	def build_FPN(self, input_tensor, istraining = True):
		with tf.variable_scope("FPN", reuse=tf.AUTO_REUSE) as scope:
			x = tf.layers.conv2d(input_tensor, 64, 7, padding = "same", strides = (2, 2))
			x = tf.layers.batch_normalization(x, training = istraining)
			x = activation(x)
			C1 = x = tf.layers.max_pooling2d(x, pool_size = 3, strides = 2, padding = "same")

			x = self._conv_block(x, [64, 64, 256], strides = (1,1), istraining = istraining)
			x = self._identity_block(x, [64, 64, 256], istraining = istraining)
			C2 = x = self._identity_block(x, [64, 64, 256], istraining = istraining)

			x = self._conv_block(x, [128, 128, 512], istraining = istraining)
			x = self._identity_block(x, [128, 128, 512], istraining = istraining)
			x = self._identity_block(x, [128, 128, 512], istraining = istraining)
			C3 = x = self._identity_block(x, [128, 128, 512], istraining = istraining)

			x = self._conv_block(x, [256, 256, 1024], istraining = istraining)
			for i in range(5):#resnet50
				x = self._identity_block(x, [256, 256, 1024], istraining = istraining)
			C4 = x

			x = self._conv_block(x, [512, 512, 2048], istraining = istraining)
			x = self._identity_block(x, [512, 512, 2048], istraining = istraining)
			C5 = x = self._identity_block(x, [512, 512, 2048], istraining = istraining)

			x = tf.layers.conv2d(x, filters = 256, kernel_size = 3, strides = 2, padding = "same", istraining = istraining)
			C6 = x = tf.layers.batch_normalization(x, training = istraining)

			x = tf.layers.conv2d(x, filters = 256, kernel_size = 3, strides = 2, padding = "same", istraining = istraining)
			C7 = x = tf.layers.batch_normalization(x, training = istraining)

			l3 = tf.layers.conv2d(C3, filters = 256, kernel_size = 1, strides = 1, padding = "same", istraining = istraining)
			l4 = tf.layers.conv2d(C4, filters = 256, kernel_size = 1, strides = 1, padding = "same", istraining = istraining)
			C5 = tf.layers.conv2d(C5, filters = 256, kernel_size = 1, strides = 1, padding = "same", istraining = istraining)

			shape = tf.shape(l4)
			x = l4 + tf.image.resize_nearest_neighbor(C5, (shape[1], shape[2]))
			x = tf.layers.conv2d(x, filters = 256, kernel_size = 3, strides = 1, padding = "same")
			C4 = tf.layers.batch_normalization(x, training = istraining)

			shape = tf.shape(l3)
			x = l3 + tf.image.resize_nearest_neighbor(l4, (shape[1], shape[2]))
			x = tf.layers.conv2d(x, filters = 256, kernel_size = 3, strides = 1, padding = "same")
			C3 = tf.layers.batch_normalization(x, training = istraining)

			return C3, C4, C5, C6, C7

	def _conv_block(self, input_tensor, filters, strides = (2,2), istraining = True):
		x = tf.layers.conv2d(input_tensor, filters[0], 1, padding = "same", strides = strides)
		x = tf.layers.batch_normalization(x, training = istraining)
		x = activation(x)

		x = tf.layers.conv2d(x, filters[1], 3, padding = "same")
		x = tf.layers.batch_normalization(x, training = istraining)
		x = activation(x)

		x = tf.layers.conv2d(x, filters[2], 1, padding = "same")
		x = tf.layers.batch_normalization(x, training = istraining)

		shortcut = tf.layers.conv2d(input_tensor, filters[2], 1, padding = "same", strides = strides)
		shortcut = tf.layers.batch_normalization(shortcut, training = istraining)

		x = tf.add(x, shortcut)
		x = activation(x)
		return x

	def _identity_block(self, input_tensor, filters, istraining = True):
		x = tf.layers.conv2d(input_tensor, filters[0], 1, padding = "same")
		x = tf.layers.batch_normalization(x, training = istraining)
		x = activation(x)

		x = tf.layers.conv2d(x, filters[1], 3, padding = "same")
		x = tf.layers.batch_normalization(x, training = istraining)
		x = activation(x)

		x = tf.layers.conv2d(x, filters[2], 1, padding = "same")
		x = tf.layers.batch_normalization(x, training = istraining)

		x = tf.add(x, input_tensor)
		x = activation(x)
		return x		

	def _class_subnet(self, x, num_anchors_per_loc, level, istraining = True):
		with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
			for i in range(4):
				x = tf.layers.conv2d(x, filters = 256, kernel_size = 3, strides = 1, padding = "same", name = "class_{}".format(i))
				x = tf.layers.batch_normalization(x, training = istraining, name = "class_{}_bn_levle_{}".format(i, level))
				x = activation(x)

			output = tf.layers.conv2d(x, filters = (self.class_num+1)*num_anchors_per_loc, kernel_size = 3, padding = "same", name = "class_output")
			return output

	def _box_subnet(self, x, num_anchors_per_loc, level, istraining = True):
		with tf.variable_scope('box_net', reuse=tf.AUTO_REUSE):
			for i in range(4):
				x = tf.layers.conv2d(x, filters = 256, kernel_size = 3, strides = 1, padding = "same", name='box_{}'.format(i))
				x = tf.layers.batch_normalization(x, training = istraining, name = "box_{}_bn_levle_{}".format(i, level))
				x = activation(x)

			output = tf.layers.conv2d(x, filters = 4*num_anchors_per_loc, kernel_size = 3, padding = "same", name = "box_output")
			return output

	def loss(self, class_pred, box_pred, class_label, box_label, pos_indices, bg_indices):
		reg_loss = reg_loss(box_pred, box_label, pos_indices)
		cls_loss = focal_loss(class_pred, class_label, pos_indices | bg_indices)

		normalizer = tf.maximum(tf.reduce_sum(tf.to_float(pos_indices)), 1.0)
		normalized_reg_loss = tf.multiply(reg_loss, 1.0/4.0/normalizer)
		normalized_cls_loss = tf.multiply(cls_loss, 1.0/normalizer)
		return normalized_cls_loss, normalized_reg_loss

	def train(self):
		box_tensor = tf.placeholder(tf.float32, (None, None, 4))
		class_tensor = tf.placeholder(tf.float32, (None, None, self.class_num + 1))
		pos_indices_tensor = tf.placeholder(tf.float32, (None, None, 1))
		bg_indices_tensor = tf.placeholder(tf.float32, (None, None, 1))

		boxUtil = BBoxUtility((256, 256), self.class_num)

		box_pred, class_pred = self.build(num_anchors_per_loc = boxUtil.num_anchors_per_loc)

		loss = self.loss(class_pred, box_pred, class_label, box_label, pos_indices, bg_indices)
		loss = tf.reduce_sum(loss)

		lr = tf.train.exponential_decay(1e-4, self.global_step, 10000 ,0.9, True, name='learning_rate')
		optimizer = tf.train.AdamOptimizer(lr)
		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			optim = optimizer.minimize(loss, global_step = self.global_step)

		dl = DataLoader(boxUtil)

		c = tf.ConfigProto()
		c.gpu_options.allow_growth = True
		with tf.Session() as sess:
			print("start session")
			sess.run(tf.global_variables_initializer())
			