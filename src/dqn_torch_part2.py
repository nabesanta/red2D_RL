#!/usr/bin/env python3
import os
import csv
import copy
import numpy as np
from PIL import Image
from collections import deque
# 強化学習環境
import redenv
import gym
# 機械学習関連torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor

# 各種設定
np.random.seed(0)
# 過去何ステップ分の状態量を使うか
STATE_NUM = 10
# 隠れ層の数
HIDDEN_SIZE = 16

print(torch.cuda.is_available())
print(torch.cuda.device_count())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Q Network Definition
class Q(nn.Module):
    def __init__(self, state_num=STATE_NUM, action=3):
        super(Q, self).__init__()
        # state: 過去の状態量の数
        # action: 行動の数
        # → 各行動に対応したQ値を出力する
        self.l1 = nn.Linear(state_num, HIDDEN_SIZE)
        self.l2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.l3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.l4 = nn.Linear(HIDDEN_SIZE, action)

    def __call__(self, x, t):
        # mse_loss(予測値, 実際の値)
        # 平均二乗損失を用いて, 予測値とターゲット値の差を計算する
        return F.mse_loss(x, t)

    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        y = F.relu(self.l4(h3))
        return y

# 経験メモリの格納場所
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)

# TD誤差の格納場所
class Memory_TDerror(Memory):
    def __init__(self, max_size=1000):
        super().__init__(max_size)
    # add, sample, len は継承されているので定義不要

# Double DQN Agent Definition
class DoubleDQNAgent():
    def __init__(self, state_num=STATE_NUM, action=3, epsilon=0.99, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.main_model = Q(state_num, action).to(device) # 行動を学習するニューラルネット
        self.target_model = Q(state_num, action).to(device) # 状態価値を学習するニューラルネット
        # ニューラルネットのパラメータを最適化する手法
        # lr: 学習率, α: 学習率を調整する係数, ε: ゼロ除算を防ぐもの
        self.optimizer = optim.RMSprop(self.main_model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        self.epsilon = epsilon # 探査率: ε-greedy法
        self.actions = [0, 1, 2] # ロボットのアクション番号
        self.experienceMemory = [] # ローカル経験メモリ
        self.experienceMemory_local = [] # グローバル経験メモリ
        self.memSize = 1000 # 1000ステップ * 10エピソード: 最初は多く探査するために多めに確保
        self.memPos = 0
        # 学習関連
        self.batch_num = 32 # 学習のバッチサイズ
        self.gamma = 0.99 # 割引率
        self.total_rewards = np.ones(10) * -1e3
        self.loss = 0
        # 経験に使用
        self.maxPosition = 0 
        self.alpha = alpha 
        self.error = 0.01
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.update_target_freq = 1 # 1エピソードごとに二つのニューラルネットを一致させる

# 行動の価値関数の獲得
    def get_action_value(self, seq):
        # 1次元の配列に変換
        x = tensor(np.hstack([seq]).astype(np.float32).reshape((1, -1))).to(device)
        # ニューラルネットで予測
        pact = self.main_model.predict(x)
        # maxs: 最大値, indices: そのインデックス
        maxs, indices = torch.max(pact.data, 1)
        # 最大値のインデックスを取り出す
        pact = indices.cpu().numpy()[0]
        return pact

    def get_greedy_action(self, seq):
        action_index = self.get_action_value(seq)
        return self.actions[action_index]

    def reduce_epsilon(self, episode):
        # エピソード0: 0.901, エピソード10: 0.082, エピソード100: 0.001
        # self.epsilon = 0.001 + 0.9 / (1.0 + episode)
        # 1000エピソードで0.01になる減少関数
        initial_epsilon = 0.99
        final_epsilon = 0.01
        decay_rate = (initial_epsilon - final_epsilon) / 100
        # 更新する際に使用する
        self.epsilon = max(final_epsilon, initial_epsilon - decay_rate * episode)

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
        for x in self.experienceMemory_local:
            self.experience(x, error=None)

    # 経験をグローバル経験メモリに追加する
    def experience(self, x, error=None):
        # 優先度の獲得
        if (error==None):
            error = 0
        # 投入するときの優先度はすべて同じ
        priority = (error + self.error) ** self.alpha
        self.experienceMemory.append((x, priority))
        # # 満タンになったら先頭から順番に消していく
        # if len(self.experienceMemory) > self.memSize:
        #     self.experienceMemory.pop(0)
        # 優先度の低いものから削除する
        if len(self.experienceMemory) > self.memSize:
            # 優先度が最も低いものを見つけて削除する
            min_priority_idx = min(range(len(self.experienceMemory)), key=lambda idx: self.experienceMemory[idx][1])
            self.experienceMemory.pop(min_priority_idx)

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
        # 優先度の更新
        for i, idx in enumerate(indices):
            if i < len(errors):
                priority = min(abs(errors[i]) + self.beta_increment_per_sampling, self.error)
                self.experienceMemory[idx] = (self.experienceMemory[idx][0], priority)

    def get_TDerror(self, state, action, reward, next_state):
        # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
        x = tensor(np.hstack([next_state]).astype(np.float32).reshape((1, -1))).to(device)
        pact = self.main_model.predict(x)
        maxs, indices = torch.max(pact.data, 1)
        pact = indices.numpy()[0]
        next_action = self.actions[pact]
        target = reward + self.gamma * self.target_model.predict(x)[pact]
        old_x = tensor(np.hstack([state]).astype(np.float32).reshape((1, -1))).to(device)
        # TD誤差の計算
        TDerror = target - self.target_model.predict(old_x)[action]
        return TDerror

    # TD誤差をすべて更新
    def update_TDerror(self, memory, gamma, mainQN, targetQN):
        for i in range(0, (self.len() - 1)):
            (state, action, reward, next_state) = memory.buffer[i]  # 最新の状態データを取り出す
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action = np.argmax(mainQN.model.predict(next_state)[0])  # 最大の報酬を返す行動を選択する
            target = reward + gamma * targetQN.model.predict(next_state)[0][next_action]
            TDerror = target - targetQN.model.predict(state)[0][action]
            self.buffer[i] = TDerror

    # TD誤差の絶対値和を取得
    def get_sum_absolute_TDerror(self):
        sum_absolute_TDerror = 0
        for i in range(0, (self.len() - 1)):
            sum_absolute_TDerror += abs(self.buffer[i]) + 0.0001  # 最新の状態データを取り出す
        return sum_absolute_TDerror

    def update_model(self, num):
        if len(self.experienceMemory) < self.batch_num:
            print("Not enough experiences")
            return
        # 抽出されたバッチとインデックス
        batch, indices = self.sample()
        # batchの中から、経験のみを抜き出す
        batch = np.array([e for (e, p) in batch])
        # 経験のすべての行に対して一個前の状態行列を取り出し、バッチサイズの次元に変換する
        x = tensor(batch[:, 0:STATE_NUM].reshape((self.batch_num, -1)).astype(np.float32)).to(device)
        new_x = tensor(batch[:, STATE_NUM+2:2*STATE_NUM+2].reshape((self.batch_num, -1)).astype(np.float32)).to(device)
        # 学習じゃないから、逆伝搬を防ぐ
        # NNから最適な行動を選択
        pact_array = self.main_model.predict(x)
        maxs, indices = torch.max(pact_array.data, 1)
        pact = indices.cpu().numpy()[0]
        indices_cpu = indices.cpu()
        next_actions = self.actions[pact]
        # NNから次の状態のQ値を取得する
        next_q_values = self.target_model.predict(new_x).cpu().data.numpy()
        # 現在のモデルのQ値を計算する
        targets = self.main_model.predict(x).clone().cpu().data.numpy()
        for i in range(self.batch_num):
            # 0~STATE_NUM-1に状態が入っているから
            # STATE_NUMは行動
            a = batch[i, STATE_NUM]
            # STATE_NUM+1が報酬
            r = batch[i, STATE_NUM + 1]
            # 終了判定
            bool_dones = batch[i, 2*STATE_NUM+2]
            ai = int((a + 1) / 2)
            new_seq = batch[i, (STATE_NUM + 2):(STATE_NUM * 2 + 2)]
            # 上で取得したnext_actionsを用いて, 別のNNで獲得したnext_q_valuesを計算
            targets[i, ai] = r + self.gamma * next_q_values[i, indices_cpu.numpy()[i]]*(not bool_dones)
        # 先ほど計算したtarget値を変換
        t = tensor(np.array(targets).reshape((self.batch_num, -1)).astype(np.float32)).to(device)
        # 勾配をクリアにする
        self.optimizer.zero_grad()
        # lossの計算をする
        # loss = nn.MSELoss()(pact_array, t)
        loss = nn.MSELoss()(pact_array, t)
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
        if self.memPos != num:
            print("check")
            self.target_model = copy.deepcopy(self.main_model)
            self.memPos = num

# Simulator（変更なし）
class Simulator:
    def __init__(self, environment, agent, TDmemory):
        # DDQN agent
        self.agent = agent
        # TD memory
        self.TDmemory = TDmemory
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
        total_reward = 0
        step = 0 # ステップ数
        done = False # ゲーム終了フラグ
        max_position = 0
        max_diff = 0
        frames = [] # フレームを保存するリスト
        # 環境のリセット
        init_position, _ = self.env.reset() 
        # 履歴行列のリセット
        self.reset_seq() 
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
            # 移動量の計算
            diff_distance = abs(current_position - init_position[0])
            if (max_diff < diff_distance):
                max_diff = diff_distance
                max_position = current_position
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
            print(step)
            if train:
                self.agent.update_model(num)
                self.agent.reduce_epsilon(num)
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
    memory_TDerror = Memory_TDerror(max_size=1000)
    sim = Simulator(env, agent, memory_TDerror)
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
