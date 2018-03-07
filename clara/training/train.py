import csv
import logging
import os
import random
import sys
import tensorflow as tf
from clara.agent.position import Position
from clara.agent.deep_q_network import DQN
from clara.agent.experience_memory import Memory
from clara.training.environment import Environment


# Data saving configs
DATA_DIRECTORY = './training'
os.mkdir(DATA_DIRECTORY)
MODEL_LOAD_PATH = None  # Set it if you want to load already trained model
MODEL_SAVE_PATH = DATA_DIRECTORY + '/dqn-model'
SAVING_FREQUENCY = 500000
TENSORBOARD_DATA_PATH = DATA_DIRECTORY + '/tensorboard'
TENSORBOARD_SAVING_FREQUENCY = 10000  # How many steps before new DQN Tensorboard training summary is saved
TRADES_FILE_PATH = DATA_DIRECTORY + '/trades.csv'

# Logging configs
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler(DATA_DIRECTORY + '/logs.log'), logging.StreamHandler(sys.stdout)])
TRAINING_LOGS_FREQUENCY = 1440  # How many steps before next training stats print
DANGEROUS_Q_DIFFERENCE = 1e-8

# DQN parameters
OUTPUTS = 3  # Three values for 3 different actions
STATE_SIZE = 200*5 + 1 + OUTPUTS  # 200 are ticks, 1 is EMA, and OUTPUTS are to represent the previous action
LAYERS_SIZES = [600, 400]

# Training hyperparameters
MEMORY_SIZE = 50000  # How many experiences to keep in the memory; 250000 ~= 4GB
PRE_TRAIN_STEPS = 25000  # How many steps before training begins, it should be at least TRAINING_BATCH_SIZE
TRAINING_BATCH_SIZE = 50  # How many experiences to use for each training step
TRAINING_FREQUENCY = 5  # How many actions before performing one training step
NUM_STEPS = 8000000  # How many steps to perform for training session
TARGET_UPDATE_FREQUENCY = 10000  # How many steps before updating target network

DISCOUNT_RATE = 0.99  # Discount factor on the future, expected Q values
LEARNING_RATE = 0.00001  # Learning rate of the DQN
START_EPS = 0.5  # Starting probability of choosing random action by the agent to explore the environment
END_EPS = 0.005  # Ending probability of choosing random action by the agent to explore the environment
ANNEALING_STEPS = 2000000  # How many steps of training to reduce START_EPS to END_EPS

# Environment parameters
EXCHANGE_TRANSACTION_FEE = 0.1  # in percentage from transaction, e.g. 0.1 means 0.1%
MARKET_INTERVAL = 'oneMin'  # On what type of market interval should agent be trained
# os env variable containing URI of database containing the preprocessed data for the simulation environment
STATES_DB_URI_ENV = 'STATES_DATA_DB_URI'

if STATES_DB_URI_ENV not in os.environ:
    raise EnvironmentError('States Data Database URI is not set under {}, '
                           'please set it before running the script again'.format(STATES_DB_URI_ENV))


class AgentTrainer(object):
    def __init__(self):
        self.dqn = DQN(STATE_SIZE, LAYERS_SIZES, OUTPUTS, LEARNING_RATE, DISCOUNT_RATE)
        self.experience_memory = Memory(MEMORY_SIZE)
        logging.info('Agent initialized')

        states_db_uri = os.environ[STATES_DB_URI_ENV]
        self.environment = Environment(MARKET_INTERVAL, states_db_uri, EXCHANGE_TRANSACTION_FEE)
        logging.info('Environment initialized')

        self.saver = tf.train.Saver()
        logging.info('Saver initialized')

        # Initialize training summaries for tensorboard
        with tf.name_scope('reward'):
            self.reward_placeholder = tf.placeholder(tf.float64, shape=(), name='reward')
            reward_summary = tf.summary.scalar('value', self.reward_placeholder)
        with tf.name_scope('q-values'):
            self.q_values_placeholder = tf.placeholder(tf.float64, shape=([3,]), name='q-values')
            step_summaries = self.summarize_vector(self.q_values_placeholder)
        self.step_summaries = tf.summary.merge(step_summaries + [reward_summary])

        # Training stats initialization
        self.epsilon = 0
        self.average_estimated_q = [0, 0, 0]
        self.average_reward = 0
        self.last_train_summaries_save = 0
        self.train_steps = 0
        self.test_steps = 0
        self.total_steps = 0

        # Logging stats initialization
        self.total_reward = 0
        self.last_total_reward = 0
        self.last_trades_so_far = 0
        self.last_average_trade_profitability = 0
        self.total_estimated_q = [0, 0, 0]
        self.last_estimated_q = [0, 0, 0]
        self.positions_count = [0, 0, 0]
        self.last_positions_count = [0, 0, 0]
        self.total_loss = 0
        self.last_loss = 0

    def train(self):
        self.log_training_start_info()
        with tf.Session() as sess, \
                tf.summary.FileWriter(TENSORBOARD_DATA_PATH + '/train', sess.graph) as train_writer, \
                tf.summary.FileWriter(TENSORBOARD_DATA_PATH + '/test', sess.graph) as test_writer, \
                open(TRADES_FILE_PATH, 'w', newline='') as trades_file:
            trades_writer = csv.writer(trades_file, delimiter=';')
            trades_writer.writerow(['Market', 'Timespan', 'Coin Price', 'Action Made'])

            logging.info('Starting training session...')
            sess.run(tf.global_variables_initializer())
            if MODEL_LOAD_PATH:
                self.load_model(sess, MODEL_LOAD_PATH)

            # Data initialization

            is_test = False
            eps_drop = (START_EPS - END_EPS) / ANNEALING_STEPS
            self.epsilon = START_EPS

            while self.total_steps < NUM_STEPS:
                self.total_steps += 1
                if self.train_steps < ANNEALING_STEPS and not is_test:
                    self.epsilon -= eps_drop

                initial_state, is_test, action, estimated_q, rewards, following_states = self.make_action(sess, trades_writer)
                for i, action in enumerate(Position):
                    self.experience_memory.add(initial_state, action.value, rewards[i], following_states[i])

                real_reward = next(reward for reward, action_value in zip(rewards, action.value) if action_value == 1)
                self.total_reward += real_reward
                if is_test:
                    self.update_test_summaries(test_writer, sess, real_reward, estimated_q)
                else:
                    self.update_train_summaries(train_writer, sess, real_reward, estimated_q)

                if self.train_steps % TRAINING_FREQUENCY == 0 and self.train_steps > PRE_TRAIN_STEPS and not is_test:
                    summary = self.update_online_dqn(sess)
                    if self.train_steps % TENSORBOARD_SAVING_FREQUENCY < TRAINING_FREQUENCY:
                        train_writer.add_summary(summary, self.train_steps)

                if self.train_steps % TARGET_UPDATE_FREQUENCY == 0:
                    self.dqn.copy_online_to_target(sess)

                if (self.train_steps + 1) % SAVING_FREQUENCY == 0:
                    logging.info('Saving model\n')
                    self.saver.save(sess, '{}/model-{}.ckpt'.format(MODEL_SAVE_PATH, self.train_steps))

                if self.total_steps % TRAINING_LOGS_FREQUENCY == 0:
                    self.log_training_stats(self.total_steps)

            self.saver.save(sess, '{}/model-{}.ckpt'.format(MODEL_SAVE_PATH, NUM_STEPS))

    def load_model(self, sess, load_path):
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(load_path)
        self.saver.restore(sess, ckpt.model_checkpoint_path)

    def make_action(self, sess, trades_writer):
        # make action
        initial_state, is_test = self.environment.get_curr_state_vector()
        action, estimated_q = self.dqn.get_online_network_output(initial_state, sess)
        self.total_estimated_q = [total_q + new_q for total_q, new_q in zip(self.total_estimated_q, estimated_q)]
        self.positions_count = [count + new_pos for count, new_pos in zip(self.positions_count, action.value)]
        if random.random() < self.epsilon:
            action = random.choice(list(Position))

        rewards, following_states = self.environment.make_action(action, trade_writer=trades_writer)
        return initial_state, is_test, action, estimated_q, rewards, following_states

    def update_test_summaries(self, test_writer, sess, reward, estimated_q):
        # Save all rewards and qs for testing
        summary = sess.run(self.step_summaries, feed_dict={
            self.reward_placeholder: reward,
            self.q_values_placeholder: estimated_q
        })
        test_writer.add_summary(summary, self.train_steps + self.test_steps)
        self.test_steps += 1

    def update_train_summaries(self, train_writer, sess, reward, estimated_q):
        # Save only averages once in a while for training
        self.test_steps = 0
        self.train_steps += 1
        if self.train_steps % TENSORBOARD_SAVING_FREQUENCY == 0:
            summary = sess.run(self.step_summaries, feed_dict={
                self.reward_placeholder: self.average_reward,
                self.q_values_placeholder: self.average_estimated_q
            })
            train_writer.add_summary(summary, self.train_steps)
            self.last_train_summaries_save = self.train_steps
            self.average_reward = 0
            self.average_estimated_q = [0, 0, 0]
        else:
            self.average_reward = self.average_reward + (reward - self.average_reward) \
                                                        / (self.train_steps - self.last_train_summaries_save)
            self.average_estimated_q = [average_q + (q - average_q) / (self.train_steps - self.last_train_summaries_save)
                                        for (average_q, q) in zip(self.average_estimated_q, estimated_q)]

    def update_online_dqn(self, sess):
        train_batch = self.experience_memory.get_samples(TRAINING_BATCH_SIZE)
        loss, summary = self.dqn.train(train_batch, sess)
        self.total_loss += loss
        return summary

    def log_training_stats(self, total_steps):
        logging.info('Step: {}'.format(total_steps))

        logging.info('Total reward so far: {}'.format(self.total_reward))
        logging.info('Average total reward: {}'.format(self.total_reward / total_steps))
        new_reward = self.total_reward - self.last_total_reward
        logging.info('Reward over the last {} steps: {}'.format(TRAINING_LOGS_FREQUENCY, new_reward))
        logging.info('Average reward over the last {} steps: {}'
                     .format(TRAINING_LOGS_FREQUENCY, new_reward / TRAINING_LOGS_FREQUENCY))
        self.last_total_reward = self.total_reward

        logging.info('Trades so far: {}'.format(self.environment.trades_so_far))
        new_trades_so_far = self.environment.trades_so_far - self.last_trades_so_far
        logging.info('Trades over last {} steps: {}'.format(TRAINING_LOGS_FREQUENCY, new_trades_so_far))
        logging.info('Average profitability over all trades: {}'
                     .format(self.environment.average_trade_profitability))
        total_profitability = self.environment.average_trade_profitability * self.environment.trades_so_far
        last_total_profitability = self.last_average_trade_profitability * self.last_trades_so_far
        new_average_profitability = (total_profitability - last_total_profitability) / \
                                    (self.environment.trades_so_far - self.last_trades_so_far + 1)
        logging.info('Average profitability over last {} trades: {}'
                     .format(new_trades_so_far, new_average_profitability))
        self.last_average_trade_profitability = self.environment.average_trade_profitability
        self.last_trades_so_far = self.environment.trades_so_far

        new_estimated_q = [total_q - last_q for total_q, last_q
                           in zip(self.total_estimated_q, self.last_estimated_q)]
        logging.info('Average total estimated Q [LONG, IDLE, SHORT]: {}'
                     .format([total_q / total_steps for total_q in self.total_estimated_q]))
        new_average_estimated_q = [new_q / TRAINING_LOGS_FREQUENCY for new_q in new_estimated_q]
        logging.info('Average estimated Q over the last {} steps: {}'
                     .format(TRAINING_LOGS_FREQUENCY, new_average_estimated_q))

        new_positions_count = [total - last for total, last
                               in zip(self.positions_count, self.last_positions_count)]
        logging.info('Total positions chosen by clara [LONG, IDLE, SHORT]: {}'.format(self.positions_count))
        logging.info('Positions chosen by clara [LONG, IDLE, SHORT] over the last {} steps : {}'
                     .format(TRAINING_LOGS_FREQUENCY, new_positions_count))
        self.last_positions_count = self.positions_count

        logging.info('Average loss so far: {}'.format(self.total_loss / (total_steps / TRAINING_FREQUENCY)))
        new_loss = self.total_loss - self.last_loss
        logging.info('Average loss over the last {} steps: {}'
                     .format(TRAINING_LOGS_FREQUENCY, new_loss / TRAINING_LOGS_FREQUENCY))
        self.last_estimated_q = self.total_estimated_q
        logging.info('Epsilon: {}\n'.format(self.epsilon))

    @staticmethod
    def log_training_start_info():
        logging.info('New training session starting')
        logging.info('Layers: {}, Pre Train Steps: {}, Memory Size: {}, Training Batch size: {}; Training Frequency {}, '
                     'Target Update Frequency: {}, Discount Rate {}, Learning Rate: {}, Start eps: {}, End eps: {}, '
                     'Annealing Steps: {}, Gradient Clip: {}, Lrelu Alpha: {}'
                     .format(LAYERS_SIZES, PRE_TRAIN_STEPS, MEMORY_SIZE, TRAINING_BATCH_SIZE, TRAINING_FREQUENCY,
                             TARGET_UPDATE_FREQUENCY, DISCOUNT_RATE, LEARNING_RATE, START_EPS, END_EPS, ANNEALING_STEPS,
                             DQN.GRADIENT_CLIP, DQN.LRELU_ALPHA))

    @staticmethod
    def summarize_vector(vector):
        summaries = []
        vector_shape = vector.get_shape().as_list()[0]
        print(vector.get_shape().as_list())
        for i in range(0, vector_shape):
            summaries.append(tf.summary.scalar(str(i), tf.gather(vector, i)))

        summaries.append(tf.summary.scalar('max', tf.reduce_max(vector)))
        summaries.append(tf.summary.scalar('mean', tf.reduce_mean(vector)))
        return summaries

if __name__ == '__main__':
    trainer = AgentTrainer()
    trainer.train()
