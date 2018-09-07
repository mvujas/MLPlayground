import tensorflow as tf
import numpy as np

def one_hot(data, num_classes):
	''' Transform data to one hot format

		For example, if num_classes = 10,
		5 will be transformed to [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
	'''
	if type(data) is not np.ndarray:
		raise ValueError('Input data must be numpy.ndarray!')
	if data.ndim != 1 and (data.ndim != 2 or data.shape[1] != 1):
		raise ValueError('Invalid shape of input data!')
	n = data.shape[0]
	result = np.zeros([n, num_classes]).astype(np.float32)
	if data.ndim == 1:
		for i in range(n):
			result[i][data[i]] = 1
	else:
		for i in range(n):
			result[i][data[i][0]] = 1
	return result

def model_fn(features, labels, mode, params):
	''' Boilerplate for building EstimatorSpec '''
	logits = params['inference_model_fn'](features, mode, params)

	predicted_classes = tf.argmax(logits, 1)
	if mode == tf.estimator.ModeKeys.PREDICT:
		predictions = {
			'probability_distribution': tf.nn.softmax(logits),
			'predicted_classes': predicted_classes
		}
		return tf.estimator.EstimatorSpec(
			mode,
			predictions=predictions)

	loss = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

	metrics = {
		'accuracy': tf.metrics.accuracy(
			labels=tf.argmax(labels, 1),
			predictions=predicted_classes)
	}
	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(
			mode,
			loss=loss,
			eval_metric_ops=metrics)

	assert mode == tf.estimator.ModeKeys.TRAIN

	optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
	train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

	return tf.estimator.EstimatorSpec(
		mode,
		loss=loss,
		train_op=train_op)

def input_fn(features, labels, batch_size, training=False, shuffle_buffer_size=1000):
	if labels is None:
		data = features
	else:
		data = (features, labels)

	dataset = tf.data.Dataset.from_tensor_slices(data)

	if training:
		if shuffle_buffer_size is not None:
			assert shuffle_buffer_size > 0, 'shuffle_buffer_size must be positive integer'
			dataset = dataset.shuffle(shuffle_buffer_size)
		dataset = dataset.repeat()

	assert batch_size is not None and batch_size > 0, 'batch_size must be positive integer'

	return dataset.batch(batch_size)