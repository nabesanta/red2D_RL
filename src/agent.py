import redenv
import gym
import numpy as np
import matplotlib.pyplot as plt

import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Q:
    def __init__(self, env):
        self.env = env
        self.env_low = self.env.observation_space.low
        self.env_high = self.env.observation_space.high
        self.env_dx = (self.env_high - self.env_low) / 40
        self.q_table = np.zeros((40, 40, 3))

    def get_status(self, _observation):
        if isinstance(_observation[0], (np.ndarray, list)):
            position = int((_observation[0][0] - self.env_low[0]) / self.env_dx[0])
            velocity = int((_observation[0][1] - self.env_low[1]) / self.env_dx[1])
        else:
            position = int((_observation[0] - self.env_low[0]) / self.env_dx[0])
            velocity = int((_observation[1] - self.env_low[1]) / self.env_dx[1])
        return position, velocity

    def policy(self, s, epsilon=0.1):
        if np.random.random() <= epsilon:
            return np.random.randint(3)
        else:
            p, v = self.get_status(s)
            if all(self.q_table[p, v] == 0):
                return np.random.randint(3)
            else:
                return np.argmax(self.q_table[p, v])

    def learn(self, time=5000, alpha=0.4, gamma=0.99):
        log = []
        for j in range(time):
            total = 0
            s = self.env.reset()
            done = False

            while not done:
                a = self.policy(s)
                next_s, reward, done, _, _ = self.env.step(a)
                total += reward

                p, v = self.get_status(next_s)
                G = reward + gamma * np.max(self.q_table[p, v])

                p, v = self.get_status(s)
                self.q_table[p, v, a] += alpha * (G - self.q_table[p, v, a])
                s = next_s

            log.append(total)
            if j % 100 == 0:
                print(f"{j} ===total reward=== : {total}")
        plt.plot(log)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Progress')
        plt.show()

    def render(self):
        s = self.env.reset()
        img = self.env.render()
        return img

# 環境の作成
env = gym.make('redenv-v0')

# エージェントの初期化と学習
agent = Q(env)
agent.learn()

# アニメーションの作成
fig, ax = plt.subplots()
imgs = []
for episode in range(10):  # 最初の10エピソード分のアニメーションを作成
    s = env.reset()
    done = False
    while not done:
        img = agent.render()
        imgs.append([ax.imshow(img)])
        a = agent.policy(s)
        s, _, done, _, _ = env.step(a)

ani = animation.ArtistAnimation(fig, imgs, interval=50, blit=True)
plt.axis('off')
plt.show()
