import numpy as np
import tensorflow as tf
from clara.agent.position import Position


class DQN(object):
    def __init__(self, state_vector_size, layer_sizes, outputs, learning_rate, discount_rate):
        self._online_state_vectors = tf.placeholder(shape=[None, state_vector_size], dtype=tf.float32)
        self._target_state_vectors = tf.placeholder(shape=[None, state_vector_size], dtype=tf.float32)
        online_output, online_weights, online_biases = \
            _create_dqn_model(self._online_state_vectors, layer_sizes, outputs)
        target_output, target_weights, target_biases = \
            _create_dqn_model(self._target_state_vectors, layer_sizes, outputs)

        action_indices = tf.argmax(online_output, 1)
        self._action_vectors = tf.one_hot(action_indices, outputs)

        online_max_q_values = tf.reduce_max(online_output, axis=1)
        self._immediate_rewards = tf.placeholder(shape=[None], dtype=tf.float32)
        next_state_target_max_q_value = tf.reduce_max(target_output, axis=1)
        target_max_q_values = self._immediate_rewards + tf.scalar_mul(discount_rate, next_state_target_max_q_value)
        td_errors = tf.square(online_max_q_values - target_max_q_values)
        loss_function = tf.reduce_sum(td_errors)
        self._train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_function)

        self._copy_online_to_target_ops = []
        for i, weights in enumerate(online_weights):
            self._copy_online_to_target_ops.append(target_weights[i].assign(weights))
        for i, bias in enumerate(online_biases):
            self._copy_online_to_target_ops.append(target_biases[i].assign(bias))

    def train(self, train_batch):
        self._train_step.run(feed_dict={
            self._online_state_vectors: np.vstack(train_batch[:, 0]),  # [:, 1] takes state that agent saw before making the action
            self._immediate_rewards: np.squeeze(train_batch[:, 2]),  # [:, 2] takes immediate reward following the action
            self._target_state_vectors: np.vstack(train_batch[:, 3])  # [:, 3] takes state following the action
        })

    def get_online_network_output(self, state):
        action_vector = self._action_vectors.eval(feed_dict={self._online_state_vectors: [state]})
        return Position(action_vector[0].tolist())

    def copy_online_to_target(self, session):
        for op in self._copy_online_to_target_ops:
            session.run(op)


def _create_dqn_model(state_vectors, layers_sizes, outputs):
    """
    Takes DQN hyperparameters and creates tensorflow model for it
    :param state_placeholder: tensorflow placeholders for input state
    :param layers_sizes: list of hidden layers sizes, length of the list indicates the number of layers, whereas each
            of the list elements indicates number of nodes in the layer
    :param outputs: number of outputs for DQN, each output corresponds to separate action
    :return: tensorflow model of DQN that can be used for training and later for prediction
    """
    weights = _initialize_random_weights(state_vectors.shape.as_list()[1], layers_sizes, outputs)
    biases = _initialize_random_biases(layers_sizes, outputs)
    return _model_output(state_vectors, weights, biases)


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


def _initialize_random_weights(state_vector_size, layers_sizes, outputs):
    weights = [_generate_random_weights([state_vector_size, layers_sizes[0]])]

    for i in range(1, len(layers_sizes)):
        weights.append(_generate_random_weights([layers_sizes[i - 1], layers_sizes[i]]))

    weights.append(_generate_random_weights([layers_sizes[-1], outputs]))
    return weights


def _generate_random_weights(size):
    return tf.Variable(tf.truncated_normal(size, stddev=0.1))


def _initialize_random_biases(layers_sizes, outputs):
    biases = [_generate_random_biases([layers_sizes[0]])]

    for i in range(1, len(layers_sizes)):
        biases.append(_generate_random_biases([layers_sizes[i]]))

    biases.append(_generate_random_biases([outputs]))
    return biases


def _generate_random_biases(size):
    return tf.Variable(tf.truncated_normal(size, stddev=0.1, mean=0.2))