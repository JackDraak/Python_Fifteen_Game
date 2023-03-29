# worker.py

from AI_trainer_controller import AI_trainer_controller


def worker_train(trainer: AI_trainer_controller, episodes: int, result_queue):
    trainer.train(episodes)
    result_queue.put(trainer.q_network.state_dict())