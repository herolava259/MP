import tensorflow as tf
from keras import Model

class SimpleCNNClassifier(Model):
    def __init__(self,
                 input_shape,
                 n_layer = 2,
                 pooling = (2,2),
                 strides = 1,
                 last_activation = 'sigmoid',
                 activation = 'relu'):
        super(SimpleCNNClassifier).__init__