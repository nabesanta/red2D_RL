"""
http://incompleteideas.net/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""
# 演算系ライブラリ
import math
import numpy as np
from typing import Optional

# 強化学習系
import gym
from gym import spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled


class RedmountainEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    # 単位[m]
    redinfo = {
        "red_body_mass": 0.5,
        "red_wheel_mass": 0.1,
        "red_tale_mass": 0.05,
    }

    def __init__(self, render_mode: Optional[str] = None, goal_velocity=0):
        # 丘の左端が-1.2
        self.min_position = -1.2
        # 丘の右端が0.6
        self.max_position = 0.6
        # 最大の速度は0.07
        self.max_speed = 0.07
        # ゴールポジションは0.5
        self.goal_position = 0.5
        # ゴールでの速度は0
        self.goal_velocity = goal_velocity

        # 車に作用する力
        self.force = 0.001
        # 重力
        self.gravity = 0.0025
        # 尻尾の摩擦
        self.friction = 0.0005

        # 位置と速度の最小値
        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        # 位置と速度の最大値
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)

        # レンダリングのモード
        # self.render_mode = render_mode
        # human
        self.render_mode = self.metadata["render_modes"][0]
        # rgb_array
        self.render_mode = self.metadata["render_modes"][1]

        # レンダリング時の各種設定
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        # 描画のパラメータ
        self.pos = 0.0
        self.scale = 1.0
        self.clearance = 10
        self.height_func = lambda x: 0.0

        # アクション3種類: [左に加速、何もしない、右に加速]
        self.action_space = spaces.Discrete(3)
        # 観測空間
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

    def step(self, action: int):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        position, velocity = self.state
        # 今の速度だから、一個前のアクションと今のポジションを参照？
        # math.cos(3 * position) * (-self.gravity): 坂に働く重力の影響
        velocity += (action-1) * self.force + math.cos(3 * position) * (-self.gravity) + math.cos(3 * position) * (-self.friction)
        # 速度は-0.07~0.07
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        # 位置は単純積分
        position += velocity
        # 位置は-1.2~0.6
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0
        # 終了判定は、ゴールの位置にたどり着き速度が0以上
        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        # 毎ステップ-1.0
        reward = -1.0
        # 状態は（位置、速度）
        self.state = (position, velocity)
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        # 初期位置を-0.6から-0.4の間に収める
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        # 速度は0
        self.state = np.array([self.np_random.uniform(low=low, high=high), 0])

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def _height(self, xs):
        return 0.45 * np.sin(3 * xs) + 0.55

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        # renderに必要なgfxdrawライブラリのインポート
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        # simulation世界の横幅
        # world_width = 0.6 - (-1.2) = 1.8
        world_width = self.max_position - self.min_position
        # 600/1.8 = 1000/3 = 333.3333…
        scale = self.screen_width / world_width
        # 描画画面のサイズ
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        # 描画画面の背景色
        self.surf.fill((255, 255, 255))
        # 初期位置
        pos = self.state[0]
        # 放物線上のx座標
        xs = np.linspace(self.min_position, self.max_position, 100)
        # 放物線上のy座標
        ys = self._height(xs)
        # レンダリングの画面に合わせてサイズアップ
        xys = list(zip((xs - self.min_position) * scale, ys * scale))

        # 画面の枠と坂のレンダリング
        pygame.draw.aalines(self.surf, points=xys, closed=False, color=(0, 0, 0))

        # 車体が円の場合
        # 車体の幅と高さを同じ値に設定する（ここでは直径として考える）
        self.bodyRadius = 15
        gfxdraw.aacircle(self.surf,
                        int((pos - self.min_position) * scale), 
                        int(self.bodyRadius+5 + self._height(pos) * scale),
                        int(self.bodyRadius),
                        (255, 0, 0))
        gfxdraw.filled_circle(self.surf,
                        int((pos - self.min_position) * scale), 
                        int(self.bodyRadius+5 + self._height(pos) * scale),
                        int(self.bodyRadius),
                        (255, 0, 0))
        
        # 尻尾書くよ
        self.wheelRadius = 20
        l, r, t, b = -self.wheelRadius+5, self.wheelRadius-5, 5, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(math.cos(3 * pos) + 0.8)
            coords.append((c[0] + (pos - self.min_position) * scale - self.wheelRadius, c[1] + self.clearance + self._height(pos) * scale))
        gfxdraw.aapolygon(self.surf, coords, (255, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (255, 0, 0))
        
        # 車輪書くよ
        self.wheelRadius = 20
        gfxdraw.aacircle(self.surf,
                        int((pos - self.min_position) * scale), 
                        int(self.wheelRadius + self._height(pos) * scale),
                        int(self.wheelRadius),
                        (0, 0, 0))
        gfxdraw.filled_circle(self.surf,
                        int((pos - self.min_position) * scale), 
                        int(self.wheelRadius + self._height(pos) * scale),
                        int(self.wheelRadius),
                        (0, 0, 0))

        flagx = int((self.goal_position - self.min_position) * scale)
        flagy1 = int(self._height(self.goal_position) * scale)
        flagy2 = flagy1 + 50
        gfxdraw.vline(self.surf, flagx, flagy1, flagy2, (0, 0, 0))

        gfxdraw.aapolygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )
        gfxdraw.filled_polygon(
            self.surf,
            [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)],
            (204, 204, 0),
        )

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def get_keys_to_action(self):
        # Control with left and right arrow keys.
        return {(): 1, (276,): 0, (275,): 2, (275, 276): 1}

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False