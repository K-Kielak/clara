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

OUTPUTS = 3  # Three values for 3 different actions
STATE_SIZE = 200*5 + 1 + OUTPUTS  # 200 are ticks, 1 is EMA, and OUTPUTS are to represent the previous action
LAYERS_SIZES = [600, 400]
MEMORY_SIZE = 50000  # How many experiences to keep in the memory; 250000 ~= 4GB

PRE_TRAIN_STEPS = 25000  # How many steps of random actions before training begins
TRAINING_BATCH_SIZE = 50  # How many experiences to use for each training step
TRAINING_FREQUENCY = 5  # How many actions before performing one training step
NUM_STEPS = 8000000  # How many steps to perform for training session
TARGET_UPDATE_FREQUENCY = 10000  # How many steps before updating target network
TRAINING_STATS_FREQUENCY = 10000  # How many steps before next training stats print

DISCOUNT_RATE = 0.9999  # Discount factor on the future, expected Q values
LEARNING_RATE = 0.001  # Learning rate of the DQN
START_EPS = 0.5  # Starting probability of choosing random action by the agent to explore the environment
END_EPS = 0.01  # Ending probability of choosing random action by the agent to explore the environment
ANNEALING_STEPS = 2000000  # How many steps of training to reduce START_EPS to END_EPS

EXCHANGE_TRANSACTION_FEE = 0.1  # in percentage from transaction, e.g. 0.1 means 0.1%
MARKET_INTERVAL = 'oneMin'  # On what type of market interval should agent be trained
# os env variable containing URI of database containing the preprocessed data for the simulation environment
STATES_DB_URI_ENV = 'STATES_DATA_DB_URI'

if STATES_DB_URI_ENV not in os.environ:
    raise EnvironmentError('States Data Database URI is not set under {}, '
                           'please set it before running the script again'.format(STATES_DB_URI_ENV))


def main():
    logging.info('New training session starting')
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

    with tf.Session() as sess:
        logging.info('Starting training session...')
        sess.run(tf.global_variables_initializer())
        # pre-training random steps to gather initial experience
        for _ in range(PRE_TRAIN_STEPS):
            initial_state = environment.get_curr_state_vector()
            action = random.choice(list(Position))
            reward, following_state = environment.make_action(action)
            experience_memory.add(initial_state, action, reward, following_state)

        logging.info('Pre train steps finished, starting proper training')
        # proper training
        total_decisions_made = 0
        last_descisions_made = 0
        total_reward = 0
        last_total_reward = 0
        last_trades_so_far = 0
        last_average_trade_profitability = 0
        total_estimated_q = [0, 0, 0]
        last_estimated_q = [0, 0, 0]
        positions_count = [0, 0, 0]
        last_positions_count = [0, 0, 0]
        eps_drop = (START_EPS - END_EPS) / ANNEALING_STEPS
        epsilon = START_EPS
        for i in range(NUM_STEPS):
            if i < ANNEALING_STEPS:
                epsilon -= eps_drop

            initial_state = environment.get_curr_state_vector()
            if random.random() < epsilon:
                action = random.choice(list(Position))
            else:
                action, estimated_q = dqn.get_online_network_output(initial_state)
                total_estimated_q = [total_q + new_q for total_q, new_q in zip(total_estimated_q, estimated_q)]
                total_decisions_made += 1
                positions_count = [count + new_pos for count, new_pos in zip(positions_count, action.value)]

            reward, following_state = environment.make_action(action)
            total_reward += reward
            experience_memory.add(initial_state, action, reward, following_state)

            # update online DQN
            if i % TRAINING_FREQUENCY == 0:
                train_batch = experience_memory.get_samples(TRAINING_BATCH_SIZE)
                dqn.train(train_batch)

            # copy online DQN parameters to the target DQN
            if i % (TRAINING_FREQUENCY * TARGET_UPDATE_FREQUENCY) == 0:
                dqn.copy_online_to_target(sess)

            # print training stats
            if i % TRAINING_STATS_FREQUENCY == 0:
                logging.info('Step (after {} pre training steps): {}'.format(PRE_TRAIN_STEPS, i))

                logging.info('Total reward so far: {}'.format(total_reward))
                logging.info('Average total reward: {}'.format(total_reward / (i + 1)))
                new_reward = total_reward - last_total_reward
                logging.info('Reward over the last {} steps: {}'.format(TRAINING_STATS_FREQUENCY, new_reward))
                logging.info('Average reward over the last {} steps: {}'.format(TRAINING_STATS_FREQUENCY,
                                                                                new_reward / TRAINING_STATS_FREQUENCY))
                last_total_reward = total_reward

                logging.info('Trades so far: {}'.format(environment.trades_so_far))
                logging.info('Trades over last {} steps: {}'.format(TRAINING_STATS_FREQUENCY,
                                                                    environment.trades_so_far - last_trades_so_far))
                logging.info('Average profitability over all trades: {}'.format(environment.average_trade_profitability))
                total_profitability = environment.average_trade_profitability * environment.trades_so_far
                last_total_profitability = last_average_trade_profitability * last_trades_so_far
                new_average_profitability = (total_profitability - last_total_profitability) / \
                                            (environment.trades_so_far - last_trades_so_far)
                logging.info('Average profitability over last {} trades: {}'
                             .format((environment.trades_so_far - last_trades_so_far), new_average_profitability))
                last_average_trade_profitability = environment.average_trade_profitability
                last_trades_so_far = environment.trades_so_far

                new_decisions_made = total_decisions_made - last_descisions_made + 1
                new_estimated_q = [total_q - last_q for total_q, last_q in zip(total_estimated_q, last_estimated_q)]
                logging.info('Average total estimated Q [LONG, IDLE, SHORT]: {}'
                             .format([total_q / (total_decisions_made + 1) for total_q in total_estimated_q]))
                logging.info('Average estimated Q over the last {} steps: {}'
                             .format(TRAINING_STATS_FREQUENCY, [new_q / new_decisions_made for new_q in new_estimated_q]))

                new_positions_count = [total - last for total, last in zip(positions_count, last_positions_count)]
                logging.info('Total positions chosen by clara [LONG, IDLE, SHORT]: {}'.format(positions_count))
                logging.info('Positions chosen by clara [LONG, IDLE, SHORT] over the last {} steps : {}'
                             .format(TRAINING_STATS_FREQUENCY, new_positions_count))
                last_positions_count = positions_count

                last_estimated_q = total_estimated_q
                last_descisions_made = total_decisions_made
                logging.info('Epsilon: {}\n'.format(epsilon))


if __name__ == '__main__':
    main()
