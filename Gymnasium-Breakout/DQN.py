from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import DQN
import gymnasium as gym
import ale_py
import time

def main():
    # Crear el entorno de Breakout con render_mode='rgb_array' para entrenar rápidamente
    env = gym.make("ALE/Breakout-v5", render_mode="human")

    # Ajustar el tamaño de la imagen (opcional para optimización)
    env = gym.wrappers.ResizeObservation(env, (84, 84))  # Reducir resolución a 84x84 píxeles

    # Crear el wrapper para manejar múltiples entornos
    env = DummyVecEnv([lambda: env])

    # Crear el modelo DQN con la política CNN
    model = DQN("CnnPolicy", env, verbose=1, buffer_size=20000)

    # Entrenamiento del modelo
    total_episodes = 1000
    for episode in range(1, total_episodes + 1):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)  # Obtener acción predicha por el modelo
            obs, reward, done, info = env.step(action)  # Realizar la acción en el entorno

        # Renderizar solo cada 100 episodios para ver el progreso
        if episode % 100 == 0:
            print(f"Mostrando renderizado en el episodio {episode}")
            env.render()  # Mostrar el renderizado
            time.sleep(0.1)  # Pausa solo cuando se renderiza

        # Actualizar el modelo después de cada episodio o cada ciertos episodios
        model.learn(total_timesteps=10)  # Entrenar por 1000 pasos adicionales en cada episodio
        print(f"El episodio {episode} ha terminado.")

    env.close()  # Cerrar el entorno al finalizar

if __name__ == "__main__":
    main()
