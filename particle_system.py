from py5canvas import *
import random as py_random


class Particle:
    def __init__(self, x, y, life):
        self.x = x
        self.y = y
        self.vx = py_random.uniform(-2, 2)  # Velocity in x direction
        self.vy = py_random.uniform(-2, 2)  # Velocity in y direction
        self.ax = py_random.uniform(-0.05, 0.05)  # Acceleration in x direction
        self.ay = py_random.uniform(-0.05, 0.05)  # Acceleration in y direction
        self.life = life
        self.max_life = life  # Keep track of the initial life for fading and size
        self.size = py_random.uniform(5, 15)  # Initial size of the particle
        self.color_start = [255, 100, 100, 255]  # RGBA starting color (red)
        self.color_end = [100, 100, 255, 0]  # RGBA ending color (blue, transparent)

    def update(self):
        # Update velocity with acceleration
        self.vx += self.ax
        self.vy += self.ay
        # Update position
        self.x += self.vx
        self.y += self.vy
        # Decrease life
        self.life -= 1

    def show(self):
        # Calculate fading and shrinking
        life_ratio = self.life / self.max_life
        current_color = [
            int(self.color_start[i] * life_ratio + self.color_end[i] * (1 - life_ratio))
            for i in range(4)
        ]
        # Set fill color with RGBA
        fill((current_color[0], current_color[1], current_color[2], current_color[3]))
        ellipse(self.x, self.y, self.size * life_ratio, self.size * life_ratio)  # Shrinking size
