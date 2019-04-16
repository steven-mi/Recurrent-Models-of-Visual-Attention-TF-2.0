import tensorflow as tf

def get_weight(shape, name, trainable=True):
    """
    This method initializes randomly the weights with a min and max value of 0.1
    
    Parameters
    ----------
    shape: List
        The shape of the weight
    name: String
        The name of the weight
    trainable: boolean (default True)
        If its False then the variables are not trainable
    
    Returns
    ----------
    weight: tf.Variable
        The weight with the given configuration    
    """
    initial = tf.random.uniform(shape, minval=-0.1, maxval = 0.1)
    #initial = tf.contrib.layers.xavier_initializer()(shape)
    return tf.Variable(initial, trainable=trainable, name=name+'_W', dtype=tf.float32)

def get_bias(shape, name, trainable=True):
    """
    This method initializes the bias with a constant value of 0
    
    Parameters
    ----------
    shape: List
        The shape of the weight
    name: String
        The name of the weight
    trainable: boolean (default True)
        If its False then the variables are not trainable
    
    Returns
    ----------
    bias: tf.Variable
        The bias with the given configuration    
    """
    return tf.Variable(tf.zeros(shape), trainable=trainable, name=name+'_b', dtype=tf.float32)

class ConvLayer():
    """
    TODO
    """
    def __init__(self, n_filter, filter_size, input_channel, name):      
        """
        This method initializes the convolution layer
            
        Parameters
        ----------
        n_filter: int
            The amount of filter which the convolution will use (alias the output channel size)
        filter_size: int
            The size of a single kernel of the convolution layer
        input_channel: int
            The channel size of the input
        name: String
            The name of the convolution layer
        """  
        self.weights = get_weight([filter_size, filter_size, input_channel, n_filter], name)
        self.biases = get_bias([n_filter], name)
        self.name = name
        
        self.batch_norm = tf.keras.layers.BatchNormalization(name=self.name + "_batch_norm")
    
    def __call__(self, x):
        """
        TODO 

        Parameters
        ----------
        x: np.array
            TODO
        
        Returns
        ----------
        TODO    
        """
        conv = tf.nn.conv2d(x, self.weights, [1, 1, 1, 1], padding='VALID', data_format="NHWC", name=self.name + "_convolution")
        bias = tf.nn.bias_add(conv, self.biases, name=self.name + "_add_bias")
        batch_norm = self.batch_norm(bias)
        relu = tf.nn.relu(batch_norm, name=self.name+"_relu")
        return relu