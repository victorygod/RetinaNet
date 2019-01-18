import tensorflow as tf

def focal_loss(logits, labels, indices, alpha = 0.25, gamma = 2.0):
	logits = tf.cast(logits, tf.float32)
	labels = tf.cast(labels, tf.float32)
	ce = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits)
	pred = tf.sigmoid(logits)
	pred_pt = tf.where(tf.equal(labels, 1), pred, 1. - pred)
	alpha_t = tf.scalar_mul(alpha, tf.ones_like(labels, dtype = tf.float32))
	alpha_t = tf.where(tf.equal(labels, 1.0), alpha_t, 1-alpha_t)
	weighted_loss = ce * tf.pow(1-pred_pt, gamma) * alpha_t * indices
	return tf.reduce_sum(weighted_loss)

def reg_loss(box_pred, box_label, pos_indices, delta = 1.0):
	return tf.losses.huber_loss(predictions = box_pred, labels = box_label, weights = pos_indices, delta = delta)