import os
import argparse
import gym
import gym_donkeycar
import time

from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

# Initialize enviroment
def make_env(env_id, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env.reset()
        return env
    set_global_seeds(seed)
    return _init


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ppo_train')
    parser.add_argument('--sim', type=str, default="C:\\Users\\Denis\\Desktop\\GymCarControl\\DonkeySimWin_Last\\donkey_sim.exe", help='path to unity simulator. maybe be left at manual if you would like to start the sim on your own.')
    parser.add_argument('--port', type=int, default=9091, help='port to use for tcp')
    parser.add_argument('--test', action="store_true", help='load the trained model and play')
    parser.add_argument('--multi', action="store_true", help='start multiple sims at once')
    parser.add_argument('--env_name', type=str, default='donkey-mountain-track-v0', help='name of donkey sim environment')

    args = parser.parse_args()

    env_id = args.env_name

    conf = {"exe_path" : args.sim,
        "host" : "127.0.0.1",
        "port" : args.port,

        "body_style" : "donkey",
        "body_rgb" : (128, 128, 128),
        "car_name" : "Denis CNN",
        "font_size" : 60,
        "max_cte" : 10,
        }


    if args.test:


        env = gym.make(args.env_name, conf=conf)
        env = DummyVecEnv([lambda: env])

        model = PPO2.load("ppo_donkey_cnn_5")

        obs = env.reset()

        rewardsum = 0
        episodelen = 0
        episodes = 0
        while episodes < 100:

            # Accuracy testing

            # Time measurement
            start = time.time()

            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()

            time_ = time.time() - start
            rewardsum = rewardsum + rewards
            episodelen = episodelen + 1

            if episodelen%50 == 0:
                print("reward: {}, length: {}".format(rewardsum, episodelen))

            if episodes < 5:
                timetest = open('timetest.txt', 'a')
                timetest.write(str(time_)+"\n")
                timetest.close()

            if dones:
                acctest = open('acctest.txt', 'a')
                acctest.write(str(rewardsum[0])+","+str(episodelen)+"\n")
                acctest.close()
                rewardsum = 0
                episodelen = 0
                episodes = episodes + 1
                print("EPISODES: {}".format(episodes))

        print("done testing")

    else:


        env = gym.make(args.env_name, conf=conf)

        # Create the vectorized environment
        env = DummyVecEnv([lambda: env])

        # In the future, you can try other types of policy model, for example LSTM
        model = PPO2(CnnPolicy, env, verbose=1)

        obs = env.reset()
        # I changed timesteps to 50000. It is about 400 episodes
        model.learn(total_timesteps=10000)



        # Save the agent
        model.save("ppo_donkey_cnn")
        print("done training")


    env.close()
