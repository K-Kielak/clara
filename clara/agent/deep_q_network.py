import tensorflow as tf


def create_dqn_model(state_placeholder, layers_sizes, actions_number):
    """
    Takes DQN hyperparameters and creates tensorflow model for it
    :param state_placeholders: tensorflow placeholders for input state
    :param layers_sizes: list of hidden layers sizes, length of the list indicates the number of layers, whereas each
            of the list elements indicates number of nodes in the layer
    :param actions_number: number of outputs for DQN, each output corresponds to separate action
    :return: tensorflow model of DQN that can be used for training and later for prediction
    """
    weights = _initialize_random_weights(state_placeholder, layers_sizes, actions_number)
    biases = _initialize_random_biases(layers_sizes, actions_number)
    return _model_output(state_placeholder, weights, biases)


def _model_output(input, weights, biases):
    if len(weights) != len(biases):
        raise RuntimeError('weight and bias collections must have the same length!')

    # calculate first layer activation
    layer_sum = tf.matmul(input, weights[0]) + biases[0]
    activation = tf.nn.relu(layer_sum)

    # calculate activations of the remaining HIDDEN layer (therefore iterate until len - 1 to leave output layer
    # as output layer will have linear output, not RELU
    for i in range(1, len(weights)-1):
        layer_sum = tf.matmul(activation, weights[i]) + biases[i]
        activation = tf.nn.relu(layer_sum)

    output = tf.matmul(activation, weights[-1]) + biases[-1]
    return output, weights, biases


def _initialize_random_weights(state_placeholder, layers_sizes, actions_number):
    weights = [_generate_random_weights([len(state_placeholder), layers_sizes[0]])]

    for i in range(1, len(layers_sizes)):
        weights.append(_generate_random_weights([layers_sizes[i - 1], layers_sizes[i]]))

    weights.append(_generate_random_weights([layers_sizes[len(layers_sizes) - 1], actions_number]))
    return weights


def _generate_random_weights(size):
    return tf.Variable(tf.truncated_normal(size, stddev=0.1))


def _initialize_random_biases(layers_sizes, actions):
    biases = [_generate_random_biases([layers_sizes[0]])]

    for i in range(1, len(layers_sizes)):
        biases.append(_generate_random_biases([layers_sizes[i]]))

    biases.append(_generate_random_biases([actions]))
    return biases


def _generate_random_biases(size):
    return tf.Variable(tf.truncated_normal(size, stddev=0.1, mean=0.2))