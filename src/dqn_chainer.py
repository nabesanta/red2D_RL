#!/usr/bin/env python3
import os
import time
import csv
import rospy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from IPython import display
# 強化学習関連
import redenv
import gym
# 機械学習関連
# chainer
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, optimizers, serializers
from chainer import Chain

# 各種設定
np.random.seed(0)
# 過去何ステップ分の状態量を使うか
STATE_NUM = 10

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())
# GeForce GTX 1080 Ti
print(torch.cuda.get_device_capability())
# (6, 1)
# Q Network Definition

class Q(Chain):
    def __init__(self, state_num=STATE_NUM):
        super(Q, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(state_num, 16)
            self.l2 = L.Linear(16, 64)
            self.l3 = L.Linear(64, 256)
            self.l4 = L.Linear(256, 1024)
            self.l5 = L.Linear(1024, 2)

    def __call__(self, x, t):
        return F.mean_squared_error(self.predict(x, train=True), t)

    def predict(self, x, train=False):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        h4 = F.leaky_relu(self.l4(h3))
        y = self.l5(h4) 
        return y

# Double DQN Agent Definition
class DoubleDQNAgent():
    def __init__(self, state_num=STATE_NUM, epsilon=0.99, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.main_model = Q(state_num)
        self.target_model = Q(state_num)
        self.optimizer = optimizers.Adam(alpha=0.001)
        self.optimizer.setup(self.main_model)
        self.epsilon = epsilon
        self.actions = [0, 1, 2]
        self.experienceMemory = []
        self.memSize = 100 * 100 # 100ステップ * 100エピソード
        self.experienceMemory_local = []
        self.memPos = 0
        self.batch_num = 32
        self.gamma = 0.9
        self.loss = 0
        self.total_rewards = np.ones(10) * -1e6
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.abs_err_upper = 1.
        self.update_target_freq = 1000

    def get_action_value(self, seq):
        x = Variable(np.hstack([seq]).astype(np.float32).reshape((1, -1)))
        return self.main_model.predict(x).array[0]

    def get_greedy_action(self, seq):
        action_index = np.argmax(self.get_action_value(seq))
        return self.actions[action_index]

    def reduce_epsilon(self):
        self.epsilon = max(0.1, self.epsilon - 1.0 / 1000)

    def get_epsilon(self):
        return self.epsilon

    def get_action(self, seq, train):
        action = 0
        if train and np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.get_greedy_action(seq)
        return action

    def experience_local(self, old_seq, action, reward, new_seq):
        self.experienceMemory_local.append(np.hstack([old_seq, action, reward, new_seq]))

    def experience_global(self, total_reward):
        if np.min(self.total_rewards) < total_reward:
            print("Update reward:", np.min(self.total_rewards), "<", total_reward)
            print("Adding experiences to memory with higher reward")
            for x in self.experienceMemory_local:
                self.experience(x, error=None)  # Assuming error is None for initial addition

        # Add experiences randomly for exploration
        if np.random.random() < 0.01:
            for x in self.experienceMemory_local:
                self.experience(x, error=None)  # Assuming error is None for initial addition

        self.experienceMemory_local = []

        # Ensure memory size does not exceed memSize
        if len(self.experienceMemory) > self.memSize:
            self.experienceMemory = self.experienceMemory[-self.memSize:]

    def experience(self, x, error=None):
        priority = self.abs_err_upper ** self.alpha
        self.experienceMemory.append((x, priority))
        if len(self.experienceMemory) > self.memSize:
            self.experienceMemory.pop(0)

    def sample(self):
        priorities = np.array([p for (e, p) in self.experienceMemory])
        prob = priorities / priorities.sum()
        indices = np.random.choice(len(self.experienceMemory), size=self.batch_num, p=prob)
        batch = [self.experienceMemory[idx] for idx in indices]
        return batch, indices

    def update_priorities(self, indices, errors):
        if not isinstance(errors, list):
            errors = [errors]  # Ensure errors is a list
        
        if len(errors) == 0 or len(indices) == 0:
            return
        
        for i, idx in enumerate(indices):
            if i < len(errors):
                priority = min(abs(errors[i]) + self.beta_increment_per_sampling, self.abs_err_upper)
                self.experienceMemory[idx] = (self.experienceMemory[idx][0], priority)

    def update_model(self, old_seq, action, reward, new_seq):
        if len(self.experienceMemory) < self.batch_num:
            print("Not enough experiences")
            return

        batch, indices = self.sample()
        batch = np.array([e for (e, p) in batch])

        x = Variable(batch[:, 0:STATE_NUM].reshape((self.batch_num, -1)).astype(np.float32))
        
        # Double DQNの計算
        with chainer.no_backprop_mode():
            next_actions = np.argmax(self.main_model.predict(x).array, axis=1)
            next_q_values = self.target_model.predict(x).array
            targets = self.main_model.predict(x).array.copy()
            for i in range(self.batch_num):
                a = batch[i, STATE_NUM]
                r = batch[i, STATE_NUM + 1]
                ai = int((a + 1) / 2)
                new_seq = batch[i, (STATE_NUM + 2):(STATE_NUM * 2 + 2)]
                targets[i, ai] = r + self.gamma * next_q_values[i, next_actions[i]]

        t = Variable(np.array(targets).reshape((self.batch_num, -1)).astype(np.float32))

        self.main_model.cleargrads()
        loss = self.main_model(x, t)
        self.loss = loss.array
        loss.backward()
        self.optimizer.update()

        errors = F.mean_squared_error(self.main_model.predict(x), t).data
        if errors is None:
            errors = []
        else:
            errors = errors.tolist()

        self.update_priorities(indices, errors)

        self.memPos += 1
        if self.memPos % self.update_target_freq == 0:
            self.target_model = self.main_model.copy()


# Simulator（変更なし）
class Simulator:
    def __init__(self, environment, agent):
        # DDQN agent
        self.agent = agent
        # 2D RED mountain 
        self.env = environment
        # 過去5ステップ
        self.num_seq = STATE_NUM
        self.reset_seq()
        self.learning_rate = 1.0 
        self.highscore = 0 
        self.log = []

    def reset_seq(self):
        self.seq = np.zeros(self.num_seq) 

    def push_seq(self, state):
        self.seq[1:self.num_seq] = self.seq[0:self.num_seq-1] 
        self.seq[0] = state

    def run(self, train=True, movie=False, enable_log=False, num=None):
        self.env.reset() 
        self.reset_seq() 
        total_reward = 0
        max_ang = 0
        # フレームを保存するリスト
        frames = []
        for i in range(1000):
            old_seq = self.seq.copy()
            action = self.agent.get_action(old_seq, train)
            observation, reward, done, _, _ = self.env.step(action)
            img = self.env.render()
            frames.append(Image.fromarray(img))
            self.env.close()
            angle = observation[0]
            if (max_ang < angle):
                max_ang = angle
            total_reward += reward
            state = observation
            self.push_seq(state[0])
            new_seq = self.seq.copy()
            self.agent.experience_local(old_seq, action, reward, new_seq)
            if enable_log:
                self.log.append(np.hstack([old_seq[0], action, reward]))
            if movie:
                display.clear_output(wait=True)
                display.display(self.env.get_svg())
                time.sleep(0.01)
        frames[0].save('output'+str(num)+'.gif', save_all=True, append_images=frames[1:], duration=40, loop=0)
        # 学習する場合、探査率εを減少させる
        self.agent.experience_global(total_reward)
        if train:
            self.agent.update_model(old_seq, action, reward, new_seq)
            self.agent.reduce_epsilon()
        if train == False:
            # GIFアニメーションの作成
            frames[0].save('output'+str(num)+'.gif', save_all=True, append_images=frames[1:], duration=40, loop=0)
        if enable_log:
            return total_reward, self.log
        return total_reward, max_ang


if __name__ == '__main__':
    env = gym.make('redenv-v0')
    print("observation space num: ", env.observation_space.shape[0])
    print("action space num: ", env.action_space.n)

    agent = DoubleDQNAgent()
    sim = Simulator(env, agent)
    model_dir = '/home/nabesanta/red2D_RL/src/model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    test_highscore = 0.0
    with open("/home/nabesanta/csv/redMountain/log.csv", "w") as fw:
        directory = '/home/nabesanta/csv/redMountain/'
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, 'reward.csv')
        for i in range(1000):
            total_reward, max_angle = sim.run(train=True, movie=False, num=i)
            with open('/home/nabesanta/csv/redMountain/reward.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([total_reward, max_angle])
            if i % 100 == 0:
                serializers.save_npz('/home/nabesanta/red2D_RL/src/model/{:06d}.model'.format(i), agent.main_model)
            if i % 10 == 0:
                total_reward, max_angle = sim.run(train=False, movie=False, num=i)
                if test_highscore < total_reward:
                    print("highscore!")
                    serializers.save_npz('/home/nabesanta/red2D_RL/src/model/{:06d}_hs.model'.format(i), agent.main_model)
                    test_highscore = total_reward
                print(i, total_reward, "epsilon:{:.2e}".format(agent.get_epsilon()), "loss:{:.2e}".format(agent.loss))
                aw = agent.total_rewards
                print("min:{},max:{}".format(np.min(aw), np.max(aw)))

                out = "{},{},{:.2e},{:.2e},{},{},{}\n".format(i, total_reward, agent.get_epsilon(), agent.loss, np.min(aw), np.max(aw), max_angle)
                fw.write(out)
                fw.flush()
