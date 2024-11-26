import gymnasium as gym
import ale_py

def main():
    # Crear el entorno de Tennis
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    obs = env.reset()

    # Ejecutar un bucle de juego con acciones aleatorias
    for _ in range(1000):
        action = env.action_space.sample()  # Acción aleatoria
        obs, reward, done, trunc, info = env.step(action)
        if done or trunc:
            env.reset()
    env.close()

if __name__ == "__main__":
    main()