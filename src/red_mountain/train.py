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
# 機械学習関連chainer
from chainer import cuda, Function, Variable, optimizers, serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

# 各種設定
np.random.seed(0)
# 過去何ステップ分の状態量を使うか
STATE_NUM = 10
# 隠れ層の数
HIDDEN_SIZE = 100
# Q Network Definition
class Q(Chain):
    def __init__(self, state_num=STATE_NUM, action=3, hidden_size=HIDDEN_SIZE):
        super(Q, self).__init__()
        # state: 過去の状態量の数
        # action: 行動の数
        # → 各行動に対応したQ値を出力する
        self.l1 = L.Linear(state_num, hidden_size)
        self.l2 = L.Linear(hidden_size, hidden_size)
        self.l3 = L.Linear(hidden_size, hidden_size)
        self.l4 = L.Linear(hidden_size, action)

    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        y = self.l4(h3)  # Remove ReLU from the output layer
        return y

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size):
        # 経験メモリのバッファサイズ
        self.buffer_size = buffer_size
        # 経験メモリのインデックス
        self.index = 0
        # 経験メモリのバッファ
        self.buffer = []
        # バッファサイズ分の経験メモリ
        self.priorities = np.zeros(buffer_size, dtype=np.float32)
        # 優先度の初期化
        self.priorities[0] = 1.0

    def __len__(self):
        return len(self.buffer)

    # 経験メモリに経験を追加する
    # 新しい経験の優先度は、経験メモリの最大値になる
    def push(self, experience):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer[self.index] = experience
        self.priorities[self.index] = self.priorities.max()
        self.index = (self.index + 1) % self.buffer_size # 余り

    def sample(self, batch_size, alpha=0.6, beta=0.4):
        if len(self.buffer) < self.buffer_size:
            priorities = self.priorities[:self.index]
        else:
            priorities = self.priorities[:self.buffer_size]
        # 正規化
        prob = (priorities ** alpha) / (priorities ** alpha).sum()
        # 確率に基づいてサンプリング
        indices = np.random.choice(len(prob), batch_size, p=prob)
        weights = (1 / (len(indices) * prob[indices])) ** beta
        obs, action, reward, next_obs, done = zip(*[self.buffer[i] for i in indices])
        action = np.array(action)
        reward = np.array(reward)
        done = np.array(done).astype(np.uint8)
        weights = np.array(weights, dtype=np.float32)
        weights = Variable(weights)
        return (obs, action, reward, next_obs, done, indices, weights)

    # 優先度の更新
    def update_priorities(self, indices, priorities):
        # 行ごとに優先度を取り出す
        priorities = np.sum(priorities, axis=1, keepdims=True)
        # 1e-04は優先度が0にならないようにする
        self.priorities[indices] = np.squeeze(priorities) + 1e-04

# Double DQN Agent Definition
class DoubleDQNAgent():
    def __init__(self, state_num=STATE_NUM, action=3, epsilon=0.99, alpha=0.4, beta=0.4, beta_increment_per_sampling=0.001):
        print("Model Building")
        self.main_model = Q(state_num, action) # 行動を学習するニューラルネット
        self.target_model = copy.deepcopy(self.main_model) # 状態価値を学習するニューラルネット
        
        # ニューラルネットのパラメータを最適化する手法, 勾配が難しい問題や学習が不安定な場合、逐次学習などに応用できる
        # lr: 学習率, α: 学習率を調整する係数, ε: ゼロ除算を防ぐもの
        print("Initizlizing Optimizer")
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.01)
        self.optimizer.setup(self.main_model)
        
        # 経験関連
        self.batch_num = 32 # 学習のバッチサイズ
        self.initial_exploration = 10000  # 1000ステップ * 10エピソード: Initial exploratoin. original: 5x10^4
        self.memSize = 10000 * 5 # 1000ステップ * 100エピソード: 最初は多く探査するために多めに確保

        self.epsilon = epsilon # 探査率: ε-greedy法
        self.actions = [0, 1, 2] # ロボットのアクション番号
        self.experienceMemory = [] # 経験メモリ
        self.update_target_frequency = 10000 # ターゲットのニューラルネットの更新頻度
        # 学習関連
        self.gamma = 0.99 # 割引率
        self.total_rewards = np.ones(10) * -1e3 # 記録の良い報酬値の保管場所
        self.loss = 0 # 損失を記録する際に必要なもの
        # 経験に使用
        self.replay_flag = 0
        self.rb = PrioritizedReplayBuffer(self.memSize)
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling

# 行動の価値関数の獲得
    def get_action_value(self, seq):
        # 1次元の配列に変換
        x = Variable(np.hstack([seq]).astype(np.float32).reshape((1, -1)))
        # ニューラルネットで予測
        pact = self.main_model.predict(x)
        # maxs: 最大値, indices: そのインデックス
        # 1は次元を表す。xの出力は3次元なので、その中で最大値とそのインデックスを取得
        pact = np.argmax(pact.array[0])
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

    def get_loss(self, time):
        if self.initial_exploration < time:
            # 抽出されたバッチとインデックス、重み
            x, s_action, s_reward, new_x, s_done, indices_sample, weights = \
            self.rb.sample(self.batch_num, beta = self.beta)
            # タプルをNumPy配列に変換
            x = np.array(x)
            # 形状を変換してfloat32にキャスト
            x = x.reshape((self.batch_num, -1)).astype(np.float32)
            # ChainerのVariableに変換し、勾配計算を有効にする
            x = Variable(x, requires_grad=True)
            
            # タプルをNumPy配列に変換
            new_x = np.array(new_x)
            # 形状を変換してfloat32にキャスト
            new_x = new_x.reshape((self.batch_num, -1)).astype(np.float32)
            # ChainerのVariableに変換し、勾配計算を有効にする
            new_x = Variable(new_x, requires_grad=False)
            
            # 1, メインネットワークからQ値を取得
            model_array = self.main_model.predict(x).array 
            targets = self.main_model.predict(x).array 
            # 2, メインネットワークに次状態を入力して、目標値(target)の計算で利用するの行動を取得 
            pact_array_next = self.target_model.predict(new_x)
            pact_array_next = np.argmax(pact_array_next.array[0])
            # 3, ターゲットネットワークに次状態を入力して、目標値(target)の計算で利用するQ値を取得
            next_q_values = self.target_model.predict(new_x).array 
            # 4, メインネットワークからQ値をコピーし、
            for i in range(self.batch_num):
                # 0~STATE_NUM-1に状態が入っているから
                # STATE_NUMは行動
                a = s_action[i]
                # STATE_NUM+1が報酬
                r = s_reward[i]
                # 終了判定
                bool_dones = s_done[i]
                # 上で取得したnext_actionsを用いて, 別のNNで獲得したnext_q_valuesを計算
                targets[i, int(a):int(a+1)] = r + self.gamma * next_q_values[i, int(pact_array_next):int(pact_array_next+1)] * (1 - bool_dones)
            # 形状を変換してfloat32にキャスト
            targets = targets.reshape((self.batch_num, -1)).astype(np.float32)
            # ChainerのVariableに変換し、勾配計算を有効にする
            target = Variable(targets, requires_grad=False)
            
            # TD-error clipping
            td = target - model_array  # TD error
            td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
            weights_tensor = F.expand_dims(weights, axis=1)
            td_tmp = weights_tensor * td_tmp
            td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)
            zero_val = Variable(np.zeros((self.batch_num, 3), dtype=np.float32))
            # 損失計算
            loss = F.mean_squared_error(td_clip, zero_val)
            
            # 勾配をクリアにする
            self.main_model.cleargrads()
            # TD誤差に重みづけ
            loss.backward()
            self.optimizer.update()
            
            # 経験の優先度付けをするため、エラーの計算をする
            errors = F.absolute(target - model_array)
            errors_array = errors.array  
            # 優先度の更新
            self.rb.update_priorities(indices_sample,errors_array)
            return loss.array
        else:
            return 1.0

    # モデルの更新と経験の更新
    def update_model(self, num, step, done):
        # 経験再生
        loss = self.get_loss(step)
        # Q値の更新
        if (loss == 0.0):
            return False
        if (step != 0 and done == True):
            print("update")
            print(f'Loss: {loss}')
            # lossの記録
            with open('/home/nabesanta/csv/redMountain/loss.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([loss])
        if np.mod(step, self.update_target_frequency) == 0:
            self.target_model = copy.deepcopy(self.main_model)
        return True

# Simulator（変更なし）
class Simulator:
    def __init__(self, environment, agent):
        # DDQN agent
        self.agent = agent
        # 2D RED mountain 
        self.env = environment
        # 最高ステップ数
        self.maxStep = 10000
        # レンダリング
        self.Monitor = False
        # 過去5ステップ
        self.num_seq = STATE_NUM
        # 状態履歴配列の初期化
        self.reset_seq()
        self.log = []
        # 総合ステップ
        self.total_step = 0

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
        while not done:
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
            state_seq = [old_seq, action, reward, new_seq, done]
            # 経験メモリに追加
            self.agent.rb.push(state_seq)
            # 学習するなら、モデルの更新を行う
            if train:
                # モデルの更新
                train_done = self.agent.update_model(num, self.total_step, done)
                # 1エピソードごとに減少させていく
                self.agent.reduce_epsilon(num)
                train = train_done
            if enable_log:
                self.log.append(np.hstack([old_seq[0], action, reward]))
            if (num % 10 == 0) and movie:
                img = self.env.render()
                frames.append(Image.fromarray(img))
            step += 1
            self.total_step += 1
        print("done!!!!!!!!!!!!!")
        if movie and num % 10 == 0:
            # GIFアニメーションの作成
            frames[0].save('output'+str(num)+'_'+str(step)+'.gif', save_all=True, append_images=frames[1:], duration=40, loop=0)
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
