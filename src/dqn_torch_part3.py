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
# tensorflow
import tensorflow as tf
# 機械学習関連torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import tensor
# 優先度付き経験再生
from cpprb import PrioritizedReplayBuffer
# https://zenn.dev/team411/articles/9f1db350845e98
# https://qiita.com/keisuke-nakata/items/67fc9aa18227faf621a5
# https://arxiv.org/pdf/1511.05952
# https://horomary.hatenablog.com/entry/2021/02/06/013412#%E5%AE%9F%E8%A3%85%E4%BE%8B
# https://zenn.dev/ymd_h/articles/03edcaa47a3b1c

# GPUの使用可否
print(torch.cuda.is_available())
print(torch.cuda.device_count())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# 各種設定
np.random.seed(0)
# 過去何ステップ分の状態量を使うか
STATE_NUM = 10
# 隠れ層の数
HIDDEN_SIZE = 32
# Q Network Definition
class Q(nn.Module):
    def __init__(self, state_num=STATE_NUM, action=3, hidden_size=HIDDEN_SIZE):
        super(Q, self).__init__()
        # state: 過去の状態量の数
        # action: 行動の数
        # → 各行動に対応したQ値を出力する
        self.l1 = nn.Linear(state_num, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, action)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.l1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.l2.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.l3.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.l4.weight, nonlinearity='linear')

    def __call__(self, x, t):
        # mse_loss(予測値, 実際の値)
        # 平均二乗損失を用いて, 予測値とターゲット値の差を計算する
        return F.mse_loss(x, t)

    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        y = self.l4(h3)  # Remove ReLU from the output layer
        return y

# Double DQN Agent Definition
class DoubleDQNAgent():
    def __init__(self, state_num=STATE_NUM, action=3, epsilon=0.99, alpha=0.4, beta=0.4, beta_increment_per_sampling=0.001):
        self.main_model = Q(state_num, action).to(device) # 行動を学習するニューラルネット
        self.target_model = Q(state_num, action).to(device) # 状態価値を学習するニューラルネット
        # ニューラルネットのパラメータを最適化する手法, 勾配が難しい問題や学習が不安定な場合、逐次学習などに応用できる
        # lr: 学習率, α: 学習率を調整する係数, ε: ゼロ除算を防ぐもの
        self.optimizer = optim.RMSprop(self.main_model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
        self.epsilon = epsilon # 探査率: ε-greedy法
        self.actions = [0, 1, 2] # ロボットのアクション番号
        self.experienceMemory = [] # 経験メモリ
        self.memSize = 1000 * 1000 # 1000ステップ * 1000エピソード: 最初は多く探査するために多めに確保
        self.update_target_frequency = 999 # ターゲットのニューラルネットの更新頻度
        # 学習関連
        self.batch_num = 32 # 学習のバッチサイズ
        self.gamma = 0.99 # 割引率
        self.total_rewards = np.ones(10) * -1e3 # 記録の良い報酬値の保管場所
        self.loss = 0 # 損失を記録する際に必要なもの
        # 経験に使用
        self.replay_flag = 0
        self.rb = PrioritizedReplayBuffer(self.memSize, env_dict ={"state": {"shape": 23}}, alpha = alpha)
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

# 行動の価値関数の獲得
    def get_action_value(self, seq):
        # 1次元の配列に変換
        x = tensor(np.hstack([seq]).astype(np.float32).reshape((1, -1))).to(device)
        # ニューラルネットで予測
        pact = self.main_model.predict(x)
        # maxs: 最大値, indices: そのインデックス
        # 1は次元を表す。xの出力は3次元なので、その中で最大値とそのインデックスを取得
        maxs, indices = torch.max(pact.data, 1)
        # 最大値のインデックスを取り出す
        pact = indices.cpu().numpy()[0]
        return pact

    # 価値の最大となる行動を獲得
    def get_greedy_action(self, seq):
        action_index = self.get_action_value(seq)
        return self.actions[action_index]

    def reduce_epsilon(self, episode):
        # end_episodeで0.01になる減少関数
        initial_epsilon = 0.99
        final_epsilon = 0.01
        end_episode = 1000
        decay_rate = (initial_epsilon - final_epsilon) / end_episode
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

    def get_TDerror(self, state, action, reward, next_state, batch):
        # main_modelでnext_stateのQ値を予測
        pact = self.main_model.predict(next_state)
        maxs, indices = torch.max(pact.data, 1)
        indices_cpu = indices.cpu().numpy()
        # target_modelでnext_stateのQ値を予測
        next_q_values = self.target_model.predict(next_state).cpu().data.numpy()
        # main_modelでnext_stateのQ値を予測ものをコピー
        targets = self.main_model.predict(next_state).clone().cpu().data.numpy()
        for i in range(self.batch_num):
            # バッチから状態、行動、報酬、終了判定を取り出す
            a = batch[i, STATE_NUM]
            r = batch[i, STATE_NUM + 1]
            bool_dones = batch[i, 2 * STATE_NUM + 2]
            # 終了状態でなければ次状態のQ値を考慮
            if not bool(bool_dones):
                index = int(indices_cpu[i])
                targets[i, int(a)] = r + self.gamma * next_q_values[i, index]
            else:
                targets[i, int(a)] = r  # 終了状態の場合、次状態のQ値は考慮しない
        # 現在の状態のQ値を取得
        current_q_values = self.target_model.predict(state).cpu().data.numpy()
        TDerror = targets[np.arange(self.batch_num), action.astype(int)] - current_q_values[np.arange(self.batch_num), action.astype(int)]
        print("TDerror:", TDerror)
        return np.abs(TDerror)

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

    # モデルの更新と経験の更新
    def update_model(self, num, step, done):
        # 何個経験が貯まったら学習を始めるか
        if num==0 and step ==999:
            self.replay_flag = 1
        # 学習開始
        if num!=0 and self.replay_flag == 1:
            # 抽出されたバッチとインデックス、重み
            batch = self.rb.sample(self.batch_num, beta = self.beta)["state"]
            indices_sample = self.rb.sample(self.batch_num, beta = self.beta)["indexes"]
            weights = self.rb.sample(self.batch_num, beta = self.beta)["weights"]
            # 経験のすべての行に対して一個前の状態行列を取り出し、バッチサイズの次元に変換する
            x = tensor(batch[:, 0:STATE_NUM].reshape((self.batch_num, -1)).astype(np.float32), requires_grad=True).to(device)
            new_x = tensor(batch[:, STATE_NUM+2:2*STATE_NUM+2].reshape((self.batch_num, -1)).astype(np.float32)).to(device)
            # 1, メインネットワークからQ値を取得
            model_array = self.main_model.predict(x)
            # 2, メインネットワークに次状態を入力して、目標値(target)の計算で利用するの行動を取得 
            pact_array_next = self.main_model.predict(new_x)
            maxs, indices_next = torch.max(pact_array_next.data, 1)
            indices_cpu = indices_next.cpu()
            # 3, ターゲットネットワークに次状態を入力して、目標値(target)の計算で利用するQ値を取得
            next_q_values = self.target_model.predict(new_x).cpu().data.numpy()
            # 4, メインネットワークからQ値をコピーし、
            targets = self.main_model.predict(x).clone().cpu().data.numpy()
            for i in range(self.batch_num):
                # 0~STATE_NUM-1に状態が入っているから
                # STATE_NUMは行動
                a = batch[i, STATE_NUM]
                # STATE_NUM+1が報酬
                r = batch[i, STATE_NUM + 1]
                # 終了判定
                bool_dones = batch[i, 2*STATE_NUM+2]
                new_seq = batch[i, (STATE_NUM + 2):(STATE_NUM * 2 + 2)]
                # 上で取得したnext_actionsを用いて, 別のNNで獲得したnext_q_valuesを計算
                targets[i, int(a):int(a+1)] = r + self.gamma * next_q_values[i, int(indices_cpu.numpy()[i]):int(indices_cpu.numpy()[i]+1)]
            # 先ほど計算したtarget値を変換
            target = tensor(np.array(targets).reshape((self.batch_num, -1)).astype(np.float32)).to(device)
            # 勾配をクリアにする
            self.optimizer.zero_grad()
            # lossの計算をする
            td_loss = nn.MSELoss(reduction='none')(model_array, target)
            # TD誤差に重みを適用（weightsはnumpy.ndarrayであると仮定）
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            weights_tensor = weights_tensor.unsqueeze(1)
            loss = torch.mean(weights_tensor * td_loss.cpu())
            # TD誤差に重みづけ
            loss.backward()
            self.optimizer.step()

            # 経験の優先度付けをするため、エラーの計算をする
            # errors = self.get_TDerror(x, batch[:, STATE_NUM], batch[:, STATE_NUM + 1], new_x, batch)
            # バッチサイズ分の経験に優先度付けをするため、各経験のエラーを計算する
            errors = td_loss.mean(dim=1).detach().cpu().numpy().flatten()
            # 優先度の更新
            self.rb.update_priorities(indices_sample,errors)
            # Q値の更新
            if (step != 0 and step % self.update_target_frequency == 0) or (step != 0 and done == True):
                print("update")
                print(f'Loss: {loss.item()}')
                # lossの記録
                with open('/home/nabesanta/csv/redMountain/loss.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([loss.item()])
                self.target_model = copy.deepcopy(self.main_model)
        else:
            return

# Simulator（変更なし）
class Simulator:
    def __init__(self, environment, agent):
        # DDQN agent
        self.agent = agent
        # 2D RED mountain 
        self.env = environment
        # 最高ステップ数
        self.maxStep = 1000
        # レンダリング
        self.Monitor = False
        # 過去5ステップ
        self.num_seq = STATE_NUM
        # 状態履歴配列の初期化
        self.reset_seq()
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
        max_position_left = 0
        max_position_right= 0
        max_diff_left = 0
        max_diff_right = 0
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
            current_velocity = observation[1]
            # 移動量の計算
            if (current_position < init_position[0]):
                diff_distance_left = abs(current_position - init_position[0])
                if (max_diff_left < diff_distance_left):
                    max_diff_left = diff_distance_left
                    max_position_left = current_position
            else:
                diff_distance_right = abs(current_position - init_position[0])
                if (max_diff_right < diff_distance_right):
                    max_diff_right = diff_distance_right
                    max_position_right = current_position
            # 報酬の加算
            total_reward += reward
            # observation: [position, velocity]
            state = observation
            # 今回の観測値(位置)を状態行列に追加する
            self.push_seq(state[0])
            # 新しい状態行列を作成
            new_seq = self.seq.copy()
            # 状態配列作成
            state_seq = np.hstack([old_seq, action, reward, new_seq, done])
            # 経験メモリに追加
            self.agent.rb.add(state=state_seq)
            # 学習するなら、モデルの更新を行う
            if train:
                # モデルの更新
                self.agent.update_model(num, step, done)
                # 1エピソードごとに減少させていく
                self.agent.reduce_epsilon(num)
            if enable_log:
                self.log.append(np.hstack([old_seq[0], action, reward]))
            if movie:
                img = self.env.render()
                frames.append(Image.fromarray(img))
            step += 1
            print(step)
            if done:
                break
        if movie and num % 10 == 0:
            # GIFアニメーションの作成
            frames[0].save('output'+str(num)+'.gif', save_all=True, append_images=frames[1:], duration=40, loop=0)
        if enable_log:
            return total_reward, self.log
        return total_reward, max_position_left, max_position_right, step

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
            total_reward, max_position_left, max_position_right, step = sim.run(train=True, movie=True, num=i)
            with open('/home/nabesanta/csv/redMountain/reward.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([i, total_reward, max_position_left, max_position_right, step])
