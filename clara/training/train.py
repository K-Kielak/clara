import os
import random
import tensorflow as tf
from clara.agent.position import Position
from clara.agent.deep_q_network import DQN
from clara.agent.experience_memory import Memory
from clara.training.environment import Environment

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
EPS = 0.1  # Probability of choosing random action by the agent to explore the environment

EXCHANGE_TRANSACTION_FEE = 0.1  # in percentage from transaction, e.g. 0.1 means 0.1%
MARKET_INTERVAL = 'oneMin'  # On what type of market interval should agent be trained
# os env variable containing URI of database containing the preprocessed data for the simulation environment
STATES_DB_URI_ENV = 'STATES_DATA_DB_URI'

if STATES_DB_URI_ENV not in os.environ:
    raise EnvironmentError('States Data Database URI is not set under {}, '
                           'please set it before running the script again'.format(STATES_DB_URI_ENV))


def main():
    # initialize agent
    dqn = DQN(STATE_SIZE, LAYERS_SIZES, OUTPUTS, LEARNING_RATE, DISCOUNT_RATE)
    experience_memory = Memory(MEMORY_SIZE)
    print('Agent initialized')

    # initialize environment
    states_db_uri = os.environ[STATES_DB_URI_ENV]
    environment = Environment(MARKET_INTERVAL, states_db_uri, EXCHANGE_TRANSACTION_FEE)
    print('Environment initialized')

    with tf.Session() as sess:
        print('Starting training session...')
        sess.run(tf.global_variables_initializer())
        # pre-training random steps to gather initial experience
        for _ in range(PRE_TRAIN_STEPS):
            initial_state = environment.get_curr_state_vector()
            action = random.choice(list(Position))
            reward, following_state = environment.make_action(action)
            experience_memory.add(initial_state, action, reward, following_state)

        print('pre train steps finished, starting proper trainig')
        # proper training
        total_reward = 0
        last_total_reward = 0
        last_trades_so_far = 0
        last_average_trade_profitability = 0
        for i in range(NUM_STEPS):
            initial_state = environment.get_curr_state_vector()
            if random.random() < EPS:
                action = random.choice(list(Position))
            else:
                action = dqn.get_online_network_output(initial_state)

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
                print('\n')
                print('Step (after {} pre training steps): {}'.format(PRE_TRAIN_STEPS, i))
                print('Total reward so far: {}'.format(total_reward))
                print('Average total reward: {}'.format(total_reward / (i + PRE_TRAIN_STEPS + 1)))
                new_reward = total_reward - last_total_reward
                print('Reward over the last {} steps: {}'.format(TRAINING_STATS_FREQUENCY, new_reward))
                print('Average reward over the last {} steps: {}'.format(TRAINING_STATS_FREQUENCY,
                                                                         new_reward / TRAINING_STATS_FREQUENCY))
                last_total_reward = total_reward

                print('Trades so far: {}'.format(environment.trades_so_far))
                print('Trades over last {} steps: {}'.format(TRAINING_STATS_FREQUENCY,
                                                             environment.trades_so_far - last_trades_so_far))
                print('Average profitability over all trades: {}'.format(environment.average_trade_profitability))
                total_profitability = environment.average_trade_profitability * environment.trades_so_far
                last_total_profitability = last_average_trade_profitability * \
                                           (environment.trades_so_far - last_trades_so_far)
                new_average_profitability = (total_profitability - last_total_profitability) / \
                                           (environment.trades_so_far - last_trades_so_far)
                print('Average profitability over last {} trades: {}'
                      .format((environment.trades_so_far - last_trades_so_far), new_average_profitability))
                last_average_trade_profitability = environment.average_trade_profitability
                last_trades_so_far = environment.trades_so_far


if __name__ == '__main__':
    main()
