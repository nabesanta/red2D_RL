#!/usr/bin/env python3
import os
import time
import csv
import copy
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
from chainer import serializers
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor

# 各種設定
np.random.seed(0)
# 過去何ステップ分の状態量を使うか
STATE_NUM = 10

print(torch.cuda.is_available())
print(torch.cuda.device_count())

# Q Network Definition
class Q(nn.Module):
    def __init__(self, state_num=STATE_NUM, action=None):
        super(Q, self).__init__()
        self.l1 = nn.Linear(state_num, 16)
        self.l2 = nn.Linear(16, 64)
        self.l3 = nn.Linear(64, 256)
        self.l4 = nn.Linear(256, 1024)
        self.l5 = nn.Linear(1024, action)

    def __call__(self, x, t):
        return F.mse_loss(self.predict(x, train=True), t)

    def predict(self, x, train=False):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        y = self.l5(h4) 
        return y

# Double DQN Agent Definition
class DoubleDQNAgent():
    def __init__(self, state_num=STATE_NUM, action=None, epsilon=1.00, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.main_model = Q(state_num, action)
        self.target_model = Q(state_num, action)
        self.optimizer = optim.RMSprop(self.main_model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        self.epsilon = epsilon
        self.actions = [0, 1, 2]
        self.experienceMemory = []
        self.experienceMemory_local = []
        self.memSize = 1000 # 1000ステップ * 100エピソード
        self.memPos = 0
        # 学習関連
        self.batch_num = 100
        self.gamma = 0.9
        self.loss = 0
        # self.total_rewards = np.ones(10) * -1e6
        self.total_rewards = np.ones(10) * -1e3
        # 経験に使用
        self.maxPosition = 0 
        self.alpha = alpha
        self.error = 0.01
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.update_target_freq = 1

# 行動の価値関数の獲得
    def get_action_value(self, seq):
        x = tensor(np.hstack([seq]).astype(np.float32).reshape((1, -1)))
        pact = self.main_model.predict(x)
        maxs, indices = torch.max(pact.data, 1)
        pact = indices.numpy()[0]
        return pact

    def get_greedy_action(self, seq):
        action_index = self.get_action_value(seq)
        return self.actions[action_index]

    def reduce_epsilon(self):
        self.epsilon = max(0.1, self.epsilon - 1.0 / 1000)

    def get_epsilon(self):
        return self.epsilon

    def get_action(self, seq, train):
        action = 0
        # 学習中かつnp.random.random()が探査率より小さい場合
        if train and np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        # モデル評価中かつnp.random.random()が探査率より大きい場合
        else:
            action = self.get_greedy_action(np.array(seq))
        return action

# 経験の価値関数の獲得
    def experience_local(self, old_seq, action, reward, new_seq, done):
        self.experienceMemory_local.append(np.hstack([old_seq, action, reward, new_seq, done]))

    def experience_global(self, total_reward, max_diff):
        print("max_diff: "+str(max_diff))
        if (total_reward == -1000 and self.maxPosition < max_diff):
            self.maxPosition = max_diff
            print("self.maxPosition: "+str(self.maxPosition))
            for x in self.experienceMemory_local:
                self.experience(x, error=None)
        # さっき実行したエピソードの報酬が設定した報酬より少なかったら
        elif np.min(self.total_rewards) < total_reward:
            print("Update reward:", np.min(self.total_rewards), "<", total_reward)
            print("Adding experiences to memory with higher reward")
            # 最小値のインデックスを取得
            min_index = np.argmin(self.total_rewards)
            # 最小値を置き換える
            self.total_rewards[min_index] = total_reward
            # 先ほどの経験を経験メモリに追加する
            for x in self.experienceMemory_local:
                self.experience(x, error=None)
        else:
            return

        # Add experiences randomly for exploration
        if np.random.random() < 0.01:
            for x in self.experienceMemory_local:
                self.experience(x, error=None)

        # ローカルの経験メモリをスタッシュ
        self.experienceMemory_local = []

    # 経験をグローバル経験メモリに追加する
    def experience(self, x, error=None):
        # 優先度の獲得
        if (error==None):
            error = 0
        priority = (error + self.error) ** self.alpha
        self.experienceMemory.append((x, priority))
        if len(self.experienceMemory) > self.memSize:
            self.experienceMemory.pop(0)

    def sample(self):
        # すべての経験についている優先度のみを抽出
        priorities = np.array([p for (e, p) in self.experienceMemory])
        # 各経験の優先度を、全体の優先度の合計で割る＝優先度が高いものほど大きい値
        prob = priorities / priorities.sum()
        # 32個数分の経験を抽出する
        indices = np.random.choice(len(self.experienceMemory), size=self.batch_num, p=prob)
        # 対応するインデックスの経験をbatchで格納
        batch = [self.experienceMemory[idx] for idx in indices]
        # バッチと番号を変換
        return batch, indices

    def update_priorities(self, indices, errors):
        # エラーがリストでなかった場合、リストに変換
        if not isinstance(errors, list):
            errors = [errors]  # Ensure errors is a list
        # エラーがからの場合はリターン
        if len(errors) == 0 or len(indices) == 0:
            return
        
        for i, idx in enumerate(indices):
            if i < len(errors):
                priority = min(abs(errors[i]) + self.beta_increment_per_sampling, self.error)
                self.experienceMemory[idx] = (self.experienceMemory[idx][0], priority)

    def update_model(self, old_seq, action, reward, new_seq):
        if len(self.experienceMemory) < self.batch_num:
            print("Not enough experiences")
            return
        # 抽出されたバッチとインデックス
        batch, indices = self.sample()
        # batchの中から、経験のみを抜き出す
        batch = np.array([e for (e, p) in batch])
        # 経験のすべての行に対して状態行列を取り出し、バッチサイズの次元に変換する
        x = tensor(batch[:, 0:STATE_NUM].reshape((self.batch_num, -1)).astype(np.float32))
        # 学習じゃないから、逆伝搬を防ぐ
        with torch.no_grad():
            # NNから最適な行動を選択
            pact = self.main_model.predict(x)
            maxs, indices = torch.max(pact.data, 1)
            pact = indices.numpy()[0]
            next_actions = self.actions[pact]
            # NNから次の状態のQ値を取得する
            next_q_values = self.target_model.predict(x).data.numpy()
            # 現在のモデルのQ値を計算する
            targets = self.main_model.predict(x).clone().data.numpy()
            for i in range(self.batch_num):
                # 0~STATE_NUM-1に状態が入っているから
                # STATE_NUMは行動
                a = batch[i, STATE_NUM]
                # STATE_NUM+1が報酬
                r = batch[i, STATE_NUM + 1]
                ai = int((a + 1) / 2)
                new_seq = batch[i, (STATE_NUM + 2):(STATE_NUM * 2 + 2)]
                # 上で取得したnext_actionsを用いて, 別のNNで獲得したnext_q_valuesを計算
                targets[i, ai] = r + self.gamma * next_q_values[i, indices[i]]
        # 先ほど計算したtarget値を変換
        t = tensor(np.array(targets).reshape((self.batch_num, -1)).astype(np.float32))
        # 勾配をクリアにする
        self.optimizer.zero_grad()
        # lossの計算をする
        loss = self.main_model(x, t)
        print("loss: " + str(loss))
        loss.backward()
        self.optimizer.step()

        # 経験の優先度付けをするため、エラーの計算をする
        errors = F.mse_loss(self.main_model.predict(x), t).data
        if errors is None:
            errors = []
        else:
            errors = errors.tolist()
        # 優先度の更新を行う
        self.update_priorities(indices, errors)
        # Q値の更新
        self.memPos += 1
        if self.memPos % self.update_target_freq == 0:
            self.target_model = copy.deepcopy(self.main_model)


# Simulator（変更なし）
class Simulator:
    def __init__(self, environment, agent):
        # DDQN agent
        self.agent = agent
        # 最高ステップ数
        self.maxStep = 1000
        # レンダリング
        self.Monitor = True
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
        ini_posi, _ = self.env.reset() 
        self.reset_seq() 
        total_reward = 0
        max_ang = 0
        step = 0 # ステップ数
        done = False # ゲーム終了フラグ
        # 距離関係
        max_position = 0
        max_diff = 0
        # フレームを保存するリスト
        frames = []
        while not done and step < self.maxStep:
            # レンダリング
            if self.Monitor:
                img = self.env.render()
                frames.append(Image.fromarray(img))
            # 過去の状態履歴をコピー
            old_seq = self.seq.copy()
            # 行動の獲得
            action = self.agent.get_action(old_seq, train)
            # 行動により観測値と報酬、エピソード終了判定を行う
            observation, reward, done, _, _ = self.env.step(action)
            current_position = observation[0]
            diff_distance = abs(current_position - ini_posi[0])
            if (max_diff < diff_distance):
                max_diff = diff_distance
                max_position = current_position
            self.env.close()
            # 報酬の加算
            total_reward += reward
            # observation = [position, velocity]
            state = observation
            # 今回の観測値を状態行列に追加する
            self.push_seq(state[0])
            # 新しい状態行列を作成
            new_seq = self.seq.copy()
            # ローカルに経験メモリに蓄積
            self.agent.experience_local(old_seq, action, reward, new_seq, done)
            if enable_log:
                self.log.append(np.hstack([old_seq[0], action, reward]))
            if movie:
                img = self.env.render()
                frames.append(Image.fromarray(img))
            step += 1
        # 合計の報酬値を用いて、経験を絞っていく=良い経験のみ保持する
        self.agent.experience_global(total_reward, max_diff)
        if train:
            self.agent.update_model(old_seq, action, reward, new_seq)
            self.agent.reduce_epsilon()
        if movie:
            # GIFアニメーションの作成
            frames[0].save('output'+str(num)+'.gif', save_all=True, append_images=frames[1:], duration=40, loop=0)
        if enable_log:
            return total_reward, self.log
        return total_reward, max_position


if __name__ == '__main__':
    env = gym.make('redenv-v0')
    obs_num = env.observation_space.shape[0]
    acts_num = env.action_space.n
    print("observation space num: ", env.observation_space.shape[0])
    print("action space num: ", env.action_space.n)
    agent = DoubleDQNAgent(action=acts_num)
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
            total_reward, max_position = sim.run(train=True, movie=True, num=i)
            with open('/home/nabesanta/csv/redMountain/reward.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([total_reward, max_position])
            if i % 10 == 0:
                total_reward, max_position = sim.run(train=False, movie=True, num=i)
                if test_highscore < total_reward:
                    print("highscore!")
                    test_highscore = total_reward
                print(i, total_reward, "epsilon:{:.2e}".format(agent.get_epsilon()), "loss:{:.2e}".format(agent.loss))
                aw = agent.total_rewards
                print("min:{},max:{}".format(np.min(aw), np.max(aw)))

                out = "{},{},{:.2e},{:.2e},{},{},{}\n".format(i, total_reward, agent.get_epsilon(), agent.loss, np.min(aw), np.max(aw), max_position)
                fw.write(out)
                fw.flush()
