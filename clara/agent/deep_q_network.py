import numpy as np
import tensorflow as tf
from clara.agent.position import Position


class DQN(object):
    LRELU_ALPHA = 0.2
    GRADIENT_CLIP = 1.

    def __init__(self, state_vector_size, layer_sizes, outputs, learning_rate, discount_rate):
        # setting up placeholders
        self._curr_state_vectors = tf.placeholder(shape=[None, state_vector_size], dtype=tf.float64)
        self._made_action_vectors = tf.placeholder(shape=[None, outputs], dtype=tf.float64)
        self._immediate_rewards = tf.placeholder(shape=[None], dtype=tf.float64)
        self._next_state_vectors = tf.placeholder(shape=[None, state_vector_size], dtype=tf.float64)

        # creating DQNs
        online_weights, online_biases = \
            _create_dqn_model(state_vector_size, layer_sizes, outputs, name='online')
        target_weights, target_biases = \
            _create_dqn_model(state_vector_size, layer_sizes, outputs, trainable=False, name='target')

        # setting operations to copy online DQN parameters to target DQN
        self._copy_online_to_target_ops = []
        for i, weights in enumerate(online_weights):
            self._copy_online_to_target_ops.append(target_weights[i].assign(weights))
        for i, bias in enumerate(online_biases):
            self._copy_online_to_target_ops.append(target_biases[i].assign(bias))

        # modelling DQNs outputs
        self._online_curr_output = _model_output(self._curr_state_vectors, online_weights, online_biases)
        online_next_output = _model_output(self._next_state_vectors, online_weights, online_biases)
        target_next_output = _model_output(self._next_state_vectors, target_weights, target_biases)

        # finding best actions for current state
        best_action_indices = tf.argmax(self._online_curr_output, 1)
        self._best_action_vectors = tf.one_hot(best_action_indices, outputs, dtype=tf.float64)

        # updating online DQN according to target Q values (using Double DQN approach)
        online_q_values = tf.reduce_sum(tf.multiply(self._online_curr_output, self._made_action_vectors), axis=1)
        best_next_action_indices = tf.argmax(online_next_output, 1)
        best_next_action_vectors = tf.one_hot(best_next_action_indices, outputs, dtype=tf.float64)
        double_next_output = tf.reduce_sum(tf.multiply(target_next_output, best_next_action_vectors), 1)
        target_q_values = self._immediate_rewards + tf.scalar_mul(discount_rate, double_next_output)
        td_errors = tf.square(online_q_values - target_q_values)
        self.loss_function = tf.reduce_mean(td_errors)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads = optimizer.compute_gradients(self.loss_function)
        clipped_grads = [(tf.clip_by_value(grad, -DQN.GRADIENT_CLIP, DQN.GRADIENT_CLIP), var) for grad, var in grads]
        self._train_step = optimizer.apply_gradients(clipped_grads)

    def train(self, train_batch):
        loss = self.loss_function.eval(feed_dict={
            self._curr_state_vectors: np.vstack(train_batch[:, 0]),
            self._made_action_vectors: np.vstack(train_batch[:, 1]),
            self._immediate_rewards: np.squeeze(train_batch[:, 2]),
            self._next_state_vectors: np.vstack(train_batch[:, 3])
        })
        self._train_step.run(feed_dict={
            self._curr_state_vectors: np.vstack(train_batch[:, 0]),  # [:, 0] takes state that agent saw before making the action
            self._made_action_vectors: np.vstack(train_batch[:, 1]),  # [:, 1] takes actions made by the agent
            self._immediate_rewards: np.squeeze(train_batch[:, 2]),  # [:, 2] takes immediate reward following the action
            self._next_state_vectors: np.vstack(train_batch[:, 3])  # [:, 3] takes state following the action
        })

        return loss

    def get_online_network_output(self, state):
        action_vector = self._best_action_vectors.eval(feed_dict={self._curr_state_vectors: [state]})
        outputs = self._online_curr_output.eval(feed_dict={self._curr_state_vectors: [state]})
        return Position(action_vector[0].tolist()), outputs[0]

    def copy_online_to_target(self, session):
        for op in self._copy_online_to_target_ops:
            session.run(op)


def _create_dqn_model(state_vector_size, layers_sizes, outputs, trainable=True, name=None):
    """
    Takes DQN hyperparameters and creates tensorflow model for it
    :param state_placeholder: tensorflow placeholders for input state
    :param layers_sizes: list of hidden layers sizes, length of the list indicates the number of layers, whereas each
            of the list elements indicates number of nodes in the layer
    :param outputs: number of outputs for DQN, each output corresponds to separate action
    :return: tensorflow model of DQN that can be used for training and later for prediction
    """
    weights = _initialize_random_weights(state_vector_size, layers_sizes, outputs, trainable, name)
    biases = _initialize_random_biases(layers_sizes, outputs, trainable, name)
    return weights, biases


def _model_output(input, weights, biases):
    if len(weights) != len(biases):
        raise RuntimeError('weight and bias collections must have the same length!')

    # calculate first layer activation
    layer_sum = tf.matmul(input, weights[0]) + biases[0]
    activation = tf.nn.leaky_relu(layer_sum, alpha=tf.constant(DQN.LRELU_ALPHA, dtype=tf.float64))

    # calculate activations of the remaining HIDDEN layer (therefore iterate until len - 1 to leave output layer
    # as output layer will have linear output, not RELU
    for i in range(1, len(weights)-1):
        layer_sum = tf.matmul(activation, weights[i]) + biases[i]
        activation = tf.nn.leaky_relu(layer_sum, alpha=tf.constant(DQN.LRELU_ALPHA, dtype=tf.float64))

    output = tf.matmul(activation, weights[-1]) + biases[-1]
    return output


def _initialize_random_weights(state_vector_size, layers_sizes, outputs, trainable, name):
    weights = [_generate_random_weights([state_vector_size, layers_sizes[0]], trainable, name)]

    for i in range(1, len(layers_sizes)):
        weights.append(_generate_random_weights([layers_sizes[i - 1], layers_sizes[i]], trainable, name))

    weights.append(_generate_random_weights([layers_sizes[-1], outputs], trainable, name))
    return weights


def _generate_random_weights(size, trainable, name):
    return tf.Variable(tf.truncated_normal(size, stddev=0.00001, dtype=tf.float64), trainable=trainable, name=name)


def _initialize_random_biases(layers_sizes, outputs, trainable, name):
    biases = [_generate_random_biases([layers_sizes[0]], trainable, name)]

    for i in range(1, len(layers_sizes)):
        biases.append(_generate_random_biases([layers_sizes[i]], trainable, name))

    biases.append(_generate_random_biases([outputs], trainable, name))
    return biases


def _generate_random_biases(size, trainable, name):
    return tf.Variable(tf.truncated_normal(size, stddev=0.00001, mean=0.00003, dtype=tf.float64),
                       trainable=trainable, name=name, dtype=tf.float64)
