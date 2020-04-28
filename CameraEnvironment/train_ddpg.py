import sys
sys.path.append("./Pendulum-problem/pendulum_problem")
import numpy as np
import tensorflow as tf
from ddpg import DeepDeterministicPolicyGradients
from replay_buffer import ReplayBuffer
from neural_nets import ActorNet, CriticNet
from exploration import OrnsteinUhlenbeckActionNoise
from camera_environment import CameraEnvironment

MINIBATCH_SIZE = 16
MAX_EPISODES = 3000
CAMERA_FAILURE_PROB = 0.05
TAU = 0.001
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
GRADIENT_MAX_NORM = 5
BUFFER_SIZE = 1000000
DISCOUNT_FACTOR = 0.95
SEED = 42
SEEDTORUN = 5


def run_experiment(seed, postfix=""):
    kernel_init = tf.keras.initializers.glorot_normal(seed)
    environment = CameraEnvironment(CAMERA_FAILURE_PROB)
    action_size = environment.action_size
    state_size = environment.state_size
    action_bounds = [b[1] for b in environment.action_bounds]

    CRITIC_NET_STRUCTURE = [tf.keras.layers.Dense(600, kernel_initializer=kernel_init),
                            tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.ReLU(),
                            tf.keras.layers.Dense(600, kernel_initializer=kernel_init, activation=tf.nn.relu),
                            tf.keras.layers.Dense(1, kernel_initializer=kernel_init)
                            ]
    ACTOR_NET_STRUCTURE = [tf.keras.layers.Dense(600, kernel_initializer=kernel_init),
                           tf.keras.layers.BatchNormalization(),
                           tf.keras.layers.ReLU(),
                           tf.keras.layers.Dense(600, kernel_initializer=kernel_init, activation=tf.nn.relu),
                           tf.keras.layers.Dense(action_size, kernel_initializer=kernel_init, activation=tf.nn.tanh)
                           ]

    actor_net = ActorNet(ACTOR_NET_STRUCTURE, action_bounds, TAU, ACTOR_LEARNING_RATE)
    critic_net = CriticNet(CRITIC_NET_STRUCTURE, TAU, CRITIC_LEARNING_RATE, GRADIENT_MAX_NORM)

    action_noise = OrnsteinUhlenbeckActionNoise(np.zeros((action_size,)), 0.3)
    replay_buffer = ReplayBuffer(BUFFER_SIZE, seed)
    model = DeepDeterministicPolicyGradients(actor_net, critic_net, action_noise, replay_buffer, action_size,
                                             state_size, DISCOUNT_FACTOR, MINIBATCH_SIZE)

    logdir = f"logs/{postfix}"
    file_writer = tf.summary.create_file_writer(logdir)
    file_writer.set_as_default()
    rewards = np.zeros((MAX_EPISODES,))

    for i in range(MAX_EPISODES):
        state = environment.reset()
        ep_reward = 0

        while True:
            a = model.get_action(state)
            a = a.reshape((-1,))
            next_state, r, t, _ = environment.step(a)
            model.add_to_buffer(np.squeeze(state), a, r, t, np.squeeze(next_state))

            model.update()

            state = next_state.copy()
            ep_reward += r
            if t:
                break
        rewards[i] = ep_reward
        text = 'Reward: {:.2f} |'.format(ep_reward)
        tf.summary.scalar('Episode reward', data=ep_reward, step=i)
        tf.summary.text("Reward Logs", text, step=i)
        file_writer.flush()
        print("Run {} | Episode: {:d} | {}".format(postfix, i, text))

    for i in range(10):
        state = environment.reset()
        camera_pos = []
        object_pos = []
        while True:
            a = model.actor_predict(state)
            a = a.reshape((-1,))
            next_state, r, t, info = environment.step(a)
            camera_pos.append(info["cam_pos"].copy())
            object_pos.append(info["obj_pos"].copy())
            state = next_state.copy()
            if t:
                break
        camera_pos = np.concatenate(camera_pos, axis=0)
        object_pos = np.concatenate(object_pos, axis=0)
        np.savez(f"test_run_{i}_{postfix}.npz", camera_pos=camera_pos, object_pos=object_pos)

    return rewards


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.set_logical_device_configuration(gpu,
                                               [tf.config.LogicalDeviceConfiguration(memory_limit=3 * 1024)])

rewards = np.zeros((SEEDTORUN, MAX_EPISODES))
random_generator = np.random.RandomState(SEED)
for i in range(SEEDTORUN):
    seed = random_generator.randint(1000000)
    r = run_experiment(seed, f"seed_{i}")
    rewards[i, :] = r.copy()

np.savez(f"results_chance_{CAMERA_FAILURE_PROB}.npz", rewards=rewards)
