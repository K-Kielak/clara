import logging
import os
import random
import sys
import tensorflow as tf
from clara.agent.position import Position
from clara.agent.deep_q_network import DQN
from clara.agent.experience_memory import Memory
from clara.training.environment import Environment

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    handlers=[logging.FileHandler('training.log'), logging.StreamHandler(sys.stdout)])

MODEL_LOAD_PATH = None  # Set it if you want to load already trained model
MODEL_SAVE_PATH = './claradqn'
SAVING_FREQUENCY = 500000
TENSORBOARD_DATA_PATH = './tensorboard'

DANGEROUS_Q_DIFFERENCE = 1e-8

OUTPUTS = 3  # Three values for 3 different actions
STATE_SIZE = 200*5 + 1 + OUTPUTS  # 200 are ticks, 1 is EMA, and OUTPUTS are to represent the previous action
LAYERS_SIZES = [600, 400]
MEMORY_SIZE = 50000  # How many experiences to keep in the memory; 250000 ~= 4GB

PRE_TRAIN_STEPS = 25000  # How many steps before training begins, it should be at least TRAINING_BATCH_SIZE
TRAINING_BATCH_SIZE = 50  # How many experiences to use for each training step
TRAINING_FREQUENCY = 5  # How many actions before performing one training step
NUM_STEPS = 8000000  # How many steps to perform for training session
TARGET_UPDATE_FREQUENCY = 10000  # How many steps before updating target network
TRAINING_LOGS_FREQUENCY = 10000  # How many steps before next training stats print
TRAINING_SUMMARY_FREQUENCY = 10000  # How many steps before new DQN training summary is saved (to save space)

DISCOUNT_RATE = 0.99  # Discount factor on the future, expected Q values
LEARNING_RATE = 0.00001  # Learning rate of the DQN
START_EPS = 0.5  # Starting probability of choosing random action by the agent to explore the environment
END_EPS = 0.005  # Ending probability of choosing random action by the agent to explore the environment
ANNEALING_STEPS = 2000000  # How many steps of training to reduce START_EPS to END_EPS

EXCHANGE_TRANSACTION_FEE = 0.1  # in percentage from transaction, e.g. 0.1 means 0.1%
MARKET_INTERVAL = 'oneMin'  # On what type of market interval should agent be trained
# os env variable containing URI of database containing the preprocessed data for the simulation environment
STATES_DB_URI_ENV = 'STATES_DATA_DB_URI'

if STATES_DB_URI_ENV not in os.environ:
    raise EnvironmentError('States Data Database URI is not set under {}, '
                           'please set it before running the script again'.format(STATES_DB_URI_ENV))


def vector_summaries(vector):
    summaries = []
    vector_shape = vector.get_shape().as_list()[0]
    print(vector.get_shape().as_list())
    for i in range(0, vector_shape):
        summaries.append(tf.summary.scalar(str(i), tf.gather(vector, i)))

    summaries.append(tf.summary.scalar('max', tf.reduce_max(vector)))
    summaries.append(tf.summary.scalar('mean', tf.reduce_mean(vector)))
    return summaries


def main():
    logging.info('New training session starting')
    logging.info('Layers: {}, Pre Train Steps: {}, Memory Size: {}, Training Batch size: {}; Training Frequency {}, '
                 'Target Update Frequency: {}, Discount Rate {}, Learning Rate: {}, Start eps: {}, End eps: {}, '
                 'Annealing Steps: {}, Gradient Clip: {}, Lrelu Alpha: {}'
                 .format(LAYERS_SIZES, PRE_TRAIN_STEPS, MEMORY_SIZE, TRAINING_BATCH_SIZE, TRAINING_FREQUENCY,
                         TARGET_UPDATE_FREQUENCY, DISCOUNT_RATE, LEARNING_RATE, START_EPS, END_EPS, ANNEALING_STEPS,
                         DQN.GRADIENT_CLIP, DQN.LRELU_ALPHA))
    logging.info('.')
    logging.info('.')
    logging.info('.')
    # initialize agent
    dqn = DQN(STATE_SIZE, LAYERS_SIZES, OUTPUTS, LEARNING_RATE, DISCOUNT_RATE)
    experience_memory = Memory(MEMORY_SIZE)
    logging.info('Agent initialized')

    # initialize environment
    states_db_uri = os.environ[STATES_DB_URI_ENV]
    environment = Environment(MARKET_INTERVAL, states_db_uri, EXCHANGE_TRANSACTION_FEE)
    logging.info('Environment initialized')

    saver = tf.train.Saver()

    with tf.Session() as sess, tf.summary.FileWriter(TENSORBOARD_DATA_PATH + '/train', sess.graph) as train_writer, \
            tf.summary.FileWriter(TENSORBOARD_DATA_PATH + '/test', sess.graph) as test_writer:

        logging.info('Starting training session...')
        sess.run(tf.global_variables_initializer())

        if MODEL_LOAD_PATH:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(MODEL_LOAD_PATH)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # training stats and data initialization
        total_reward = 0
        last_total_reward = 0
        last_trades_so_far = 0
        last_average_trade_profitability = 0
        total_estimated_q = [0, 0, 0]
        last_estimated_q = [0, 0, 0]
        positions_count = [0, 0, 0]
        last_positions_count = [0, 0, 0]
        total_loss = 0
        last_loss = 0
        eps_drop = (START_EPS - END_EPS) / ANNEALING_STEPS
        epsilon = START_EPS

        with tf.name_scope('reward'):
            reward_placeholder = tf.placeholder(tf.float64, shape=(), name='reward')
            reward_summary = tf.summary.scalar('value', reward_placeholder)
        with tf.name_scope('q-values'):
            q_values_placeholder = tf.placeholder(tf.float64, shape=([3,]), name='q-values')
            step_summaries = vector_summaries(q_values_placeholder)
        step_summaries = tf.summary.merge(step_summaries + [reward_summary])

        # proper training
        train_step = 0
        test_step = 0
        while train_step < NUM_STEPS:
            if train_step < ANNEALING_STEPS:
                epsilon -= eps_drop

            # make action
            initial_state, is_test = environment.get_curr_state_vector()
            action, estimated_q = dqn.get_online_network_output(initial_state, sess)
            total_estimated_q = [total_q + new_q for total_q, new_q in zip(total_estimated_q, estimated_q)]
            positions_count = [count + new_pos for count, new_pos in zip(positions_count, action.value)]
            if random.random() < epsilon:
                action = random.choice(list(Position))

            reward, following_state = environment.make_action(action)
            total_reward += reward

            summary = sess.run(step_summaries, feed_dict={
                reward_placeholder: reward,
                q_values_placeholder: estimated_q
            })

            if is_test:
                test_writer.add_summary(summary, train_step + test_step)
                test_step += 1
            else:
                test_step = 0
                train_step += 1
                experience_memory.add(initial_state, action.value, reward, following_state)
                if train_step % TRAINING_SUMMARY_FREQUENCY == 0:
                    train_writer.add_summary(summary, train_step)

            _, next_estimated_q = dqn.get_online_network_output(following_state, sess)
            if abs(max(estimated_q) - max(next_estimated_q)) < DANGEROUS_Q_DIFFERENCE and train_step > PRE_TRAIN_STEPS:
                logging.info('Dangerous Q difference between states - Q1: {}; Q2: {}'
                             .format(estimated_q, next_estimated_q))

            # update online DQN
            if train_step % TRAINING_FREQUENCY == 0 and train_step > PRE_TRAIN_STEPS and not is_test:
                train_batch = experience_memory.get_samples(TRAINING_BATCH_SIZE)
                loss, summary = dqn.train(train_batch, sess)
                total_loss += loss
                # add training summary
                if train_step % TRAINING_SUMMARY_FREQUENCY < TRAINING_FREQUENCY:
                    train_writer.add_summary(summary, train_step)

            # copy online DQN parameters to the target DQN
            if train_step % TARGET_UPDATE_FREQUENCY == 0:
                dqn.copy_online_to_target(sess)

            # save model
            if (train_step + 1) % SAVING_FREQUENCY == 0:
                logging.info('Saving model\n')
                saver.save(sess, '{}/model-{}.ckpt'.format(MODEL_PATH, train_step))

            # print training stats
            if train_step % TRAINING_LOGS_FREQUENCY == 0:
                logging.info('Step: {}'.format(train_step))

                logging.info('Total reward so far: {}'.format(total_reward))
                logging.info('Average total reward: {}'.format(total_reward / (train_step + 1)))
                new_reward = total_reward - last_total_reward
                logging.info('Reward over the last {} steps: {}'.format(TRAINING_LOGS_FREQUENCY, new_reward))
                logging.info('Average reward over the last {} steps: {}'.format(TRAINING_LOGS_FREQUENCY,
                                                                                new_reward / TRAINING_LOGS_FREQUENCY))
                last_total_reward = total_reward

                logging.info('Trades so far: {}'.format(environment.trades_so_far))
                logging.info('Trades over last {} steps: {}'.format(TRAINING_LOGS_FREQUENCY,
                                                                    environment.trades_so_far - last_trades_so_far))
                logging.info('Average profitability over all trades: {}'.format(environment.average_trade_profitability))
                total_profitability = environment.average_trade_profitability * environment.trades_so_far
                last_total_profitability = last_average_trade_profitability * last_trades_so_far
                new_average_profitability = (total_profitability - last_total_profitability) / \
                                            (environment.trades_so_far - last_trades_so_far + 1)
                logging.info('Average profitability over last {} trades: {}'
                             .format((environment.trades_so_far - last_trades_so_far), new_average_profitability))
                last_average_trade_profitability = environment.average_trade_profitability
                last_trades_so_far = environment.trades_so_far

                new_estimated_q = [total_q - last_q for total_q, last_q in zip(total_estimated_q, last_estimated_q)]
                logging.info('Average total estimated Q [LONG, IDLE, SHORT]: {}'
                             .format([total_q / (train_step + 1) for total_q in total_estimated_q]))
                logging.info('Average estimated Q over the last {} steps: {}'
                             .format(TRAINING_LOGS_FREQUENCY, [new_q / TRAINING_LOGS_FREQUENCY for new_q in new_estimated_q]))

                new_positions_count = [total - last for total, last in zip(positions_count, last_positions_count)]
                logging.info('Total positions chosen by clara [LONG, IDLE, SHORT]: {}'.format(positions_count))
                logging.info('Positions chosen by clara [LONG, IDLE, SHORT] over the last {} steps : {}'
                             .format(TRAINING_LOGS_FREQUENCY, new_positions_count))
                last_positions_count = positions_count

                logging.info('Average loss so far: {}'.format(total_loss / (train_step + 1)))
                logging.info('Average loss over the last {} steps: {}'
                             .format(TRAINING_LOGS_FREQUENCY, (total_loss - last_loss) / TRAINING_LOGS_FREQUENCY))

                last_estimated_q = total_estimated_q
                logging.info('Epsilon: {}\n'.format(epsilon))

        saver.save(sess, '{}/model-{}.ckpt'.format(MODEL_PATH, NUM_STEPS))


if __name__ == '__main__':
    main()
