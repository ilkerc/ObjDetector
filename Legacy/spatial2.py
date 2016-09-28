import os

import matplotlib
import numpy as np

np.random.seed(123)
import matplotlib.pyplot as plt
import lasagne
import theano
import theano.tensor as T

conv = lasagne.layers.Conv2DLayer
pool = lasagne.layers.MaxPool2DLayer
NUM_EPOCHS = 100
BATCH_SIZE = 10
LEARNING_RATE = 0.001
GAUSSIAN_SIGMA = 0.3
DIM = 51
ORGFILE = u'/home/btek/Downloads/mitosisData/01_01_org_60.npy'
TRNSFILE = u'/home/btek/Downloads/mitosisData/01_01_rot_60.npy'


import numpy as np

def makeGaussian(size, fwhm = 3, center=None):
	""" Make a square gaussian kernel.

	size is the length of a side of the square
	fwhm is full-width-half-maximum, which
	can be thought of as an effective radius.
	"""

	x = np.arange(0, size, 1, float)
	y = x[:,np.newaxis]

	if center is None:
		x0 = y0 = size // 2
	else:
		x0 = center[0]
		y0 = center[1]

	return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def load_single_data(orgfile, trnsfile):
	#skipping multichannel input for now.
	orgfile = u'/home/btek/Dropbox/linuxshare/mitosisData/01_01_org_50.npy'
	trnsfile = u'/home/btek/Dropbox/linuxshare/mitosisData/01_01_rot_50.npy'
	org = np.load(orgfile)

	trn = np.load(trnsfile)
	trn = np.expand_dims(trn[:,:,1,:],2)
	trn_t = np.transpose(trn, [3, 2, 0, 1])
	org_n = org[:,:,2]
	org_n = np.expand_dims(org_n, 0)
	org_n = np.repeat(org_n, trn_t.shape[0], 0)
	org_n = np.expand_dims(org_n, 1)

	print "Train input samples:", trn_t.shape
	print "Train output samples:", org_n.shape

	d = dict(X_input=lasagne.utils.floatX(trn_t),
			 X_output=lasagne.utils.floatX(org_n),
			 num_examples_input=trn_t.shape[0],
			 num_examples_output=org_n.shape[0],
			 input_height=trn_t.shape[2],
			 input_width=trn_t.shape[3],
			 output_height=org_n.shape[2],
			 output_width=org_n.shape[3])

	return d


def build_test_net(input_width, input_height, batch_size=BATCH_SIZE):
	l_in = lasagne.layers.InputLayer(shape=(None, 1, input_width, input_height))
	l_enc1 = lasagne.layers.DenseLayer(l_in,
		num_units=input_width * input_height,nonlinearity=lasagne.nonlinearities.tanh,name= 'encoder1')

	l_enc2 = lasagne.layers.DenseLayer(l_enc1,
		num_units=input_width * input_height,nonlinearity=lasagne.nonlinearities.tanh, name= 'encoder2')

	l_out = lasagne.layers.ReshapeLayer(l_enc2, shape=(-1, 1, input_width, input_height))

	return l_out, l_enc1


def build_encoder(input_width, input_height, batch_size=BATCH_SIZE):
	ini = lasagne.init.HeUniform()
	l_in = lasagne.layers.InputLayer(shape=(None, 1, input_width, input_height))

	#l_decode = lasagne.layers.DenseLayer(l_in, num_units = input_width,
	#								   b=lasagne.init.Constant(0.0), W=lasagne.init.Constant(1.0),
	#								   nonlinearity = lasagne.nonlinearities.rectify, name ='first')
	#class_l2 = pool(l_encode, pool_size=(2, 2))
	# localization part
	b = np.zeros((2, 3), dtype=theano.config.floatX)
	b[0, 0] = 1
	b[1, 1] = 1
	b = b.flatten()
	loc_l1 = pool(l_in, pool_size=(2, 2))
	loc_l2 = conv(loc_l1, num_filters=20, filter_size=(5, 5), W=ini)
	loc_l3 = pool(loc_l2, pool_size=(2, 2))
	loc_l4 = conv(loc_l3, num_filters=20, filter_size=(7, 7), W=ini)
 
	loc_l5 = lasagne.layers.DenseLayer(loc_l4, num_units=50, W=lasagne.init.HeUniform('relu'))
 
	l_theta = lasagne.layers.DenseLayer(loc_l5, num_units=6, b=b, W=lasagne.init.Constant(0.0),
										nonlinearity=lasagne.nonlinearities.identity)

	print "TTransformer parameter output output shape: ", l_theta.output_shape

	# transformer
	l_trans1 = lasagne.layers.TransformerLayer(l_in, l_theta, downsample_factor=1)
	print "Transformer network output shape: ", l_trans1.output_shape

	#Hidden network

	class_l1 = conv(l_trans1,
		num_filters=5, filter_size=(5, 5),nonlinearity=lasagne.nonlinearities.rectify,W=ini)
	class_l2 = pool(class_l1, pool_size=(2, 2))
	# class_l3 = conv(
	#     class_l2,
	#     num_filters=16,
	#     filter_size=(3, 3),
	#     nonlinearity=lasagne.nonlinearities.rectify,
	#     W=ini,
	# )

	l_sca =  lasagne.layers.ScaleLayer(class_l2)
	l_decode = lasagne.layers.DenseLayer(l_sca, num_units = input_width * input_height,
									   b=lasagne.init.Constant(0.0), W=lasagne.init.Constant(1.0),
									   nonlinearity = lasagne.nonlinearities.linear, name ='last')


	l_out = lasagne.layers.ReshapeLayer(l_decode, shape=(-1, 1, input_width, input_height))

	print "Transformer network output shape: ", l_out.output_shape


	return l_out, l_trans1, l_theta, l_in


# this function must actually take many transformed RGB images and output a single image
def train_epoch(Xin, Xout, train_func):
	num_samples = Xin.shape[0]
	num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
	costs = []
	correct = 0
	for i in range(num_batches):
		idx = range(i * BATCH_SIZE, np.minimum((i + 1) * BATCH_SIZE, num_samples))
		X_batch = Xin[idx]
		Y_batch = Xout[idx]
		cost_batch, output_train = train_func(X_batch, Y_batch)
		costs += [cost_batch]

	return np.mean(costs)


def eval_epoch(X, y, eval_func):
	output_eval, transform_eval = eval_func(X)
	preds = np.argmax(output_eval, axis=-1)
	acc = np.mean(preds == y)
	return acc, transform_eval



data = load_single_data(ORGFILE, TRNSFILE)
Xinp = data['X_input']
Xout = data['X_output']
print "Input shape ", Xinp.shape


# model = build_test_net(data['input_width'], data['input_height'])

model, l_transform, l_teta, l_inp = build_encoder(data['input_width'], data['input_height'])
model_params = lasagne.layers.get_all_params(model, trainable=True)
model_params_sub = model_params[:-2]
print model_params_sub

# creating trainer function

X = T.tensor4()
Y = T.tensor4()

# training output
output_train = lasagne.layers.get_output(model, X, deterministic=False)
#output_eval = lasagne.layers.get_output(model, X, deterministic=True)
output_eval, transform_eval, teta_eval, input_eval = lasagne.layers.get_output([model,l_transform, l_teta, l_inp], X, deterministic=True)

gaussian_cost_weights = makeGaussian(Xinp.shape[2], Xinp.shape[2]*GAUSSIAN_SIGMA)
cost = T.mean(lasagne.objectives.squared_error(output_train, Y)*gaussian_cost_weights)
#updates = lasagne.updates.sgd(cost, model_params, learning_rate=LEARNING_RATE)
updates = lasagne.updates.adam(cost, model_params_sub, learning_rate=LEARNING_RATE)

train_func = theano.function([X, Y], [cost, output_train], updates=updates)
eval_func = theano.function([X], [output_eval,transform_eval, teta_eval, input_eval])

train_accs = []
try:
	for n in range(NUM_EPOCHS):
		print n
		train_cost = train_epoch(data['X_input'], data['X_output'], train_func)

		train_cost += [train_cost]

		print "Epoch {0}: Train cost {1}".format(n, train_cost)
except KeyboardInterrupt:
	pass

#theano.printing.pydotprint(train_func, outfile="logreg_pydotprint_prediction.png", var_with_name_simple=True)
ix = 10
[eval_X, eval_tr, eval_teta, eval_input] =eval_func(Xinp)
print "Transform params", eval_teta[ix]
fig = plt.figure(figsize=(5, 3))
ax1 = fig.add_subplot(141)
ax2 = fig.add_subplot(142)
ax3 = fig.add_subplot(143)
ax4 = fig.add_subplot(144)
ax1.imshow(eval_input[ix].reshape(DIM, DIM))
ax2.imshow(Xinp[ix,0,:,:])
ax2.set_title("Network input")
ax3.imshow(eval_tr[ix].squeeze())
ax3.set_title("Transfomer output")
ax2.axes.set_aspect('equal')
ax3.axes.set_aspect('equal')
ax4.axes.set_aspect('equal')
ax4.imshow(eval_X[ix].squeeze())
ax4.set_title("Network output")

plt.show(block=True)


#var = raw_input("Continue: ")


