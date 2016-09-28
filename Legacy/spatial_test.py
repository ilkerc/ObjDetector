import lasagne
import numpy as np
import theano.tensor as T
from PIL import Image
import theano
import matplotlib.pyplot as plt

height = 640
width = 480
I = Image.open("cat.jpg")
I = I.resize((height, width))
I = np.asarray(I) / 255.
I = I.astype('float32')
cmap = None
num_batch = 10
downsample = 1.3

# repeat Image 3 times to simulate batches
X_in = np.concatenate([[I.copy() for _ in range(num_batch)]])
X_sym = T.tensor4()

# Setup the network
l_in = lasagne.layers.InputLayer(X_in.shape)

# add channels dimension
l_dim = lasagne.layers.DimshuffleLayer(l_in, (0, 3, 1, 2))

# Create bias that is identity transform
b = np.zeros((2, 3), dtype='float32')
b[0, 0] = 0.5
b[1, 1] = 0.5
b[0, 1] = 0.5
b = b.flatten()[:6]

# Layer 1 is dense layer

l_loc1 = lasagne.layers.DenseLayer(
    l_dim,
    num_units=20,
    nonlinearity=None,
    W=lasagne.init.Constant(0.01)
)

# Layer 2 is dense layer

l_loc2 = lasagne.layers.DenseLayer(
    l_loc1,
    num_units=6,
    b=b,
    W=lasagne.init.Constant(0.0)
)

# Layer 3 is the Transformer Layer (Spatial Transformer)

l_trans = lasagne.layers.TransformerLayer(
    l_dim,
    l_loc2,
    downsample_factor=downsample
)

# Layer 4 is the shuffle layer ?

l_dim = lasagne.layers.DimshuffleLayer(
    l_trans,
    (0, 2, 3, 1)
)

# Output

out, theta = lasagne.layers.get_output(
    [l_dim, l_loc2],
    X_sym,
    deterministic=True
)

# Calculate the gradients

grads = T.grad(T.sum(out), lasagne.layers.get_all_params(l_trans))
print grads

f = theano.function([X_sym], [out, theta])
output, A = f(X_in)


# Print outputs

print "Expected output_train shape", l_dim.output_shape
print "Actual output_train shape", output.shape
print "Transformation matrix:"
print A
print "-"*20
print "Output dtype:", output.dtype


plt.figure()
plt.imshow(output[4], cmap=cmap, interpolation='none')
plt.show()
