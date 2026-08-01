import random

import wandb

run = wandb.init(project="ml-journey", name="phase0-smoke-test")

loss = 2.0
for step in range(20):
    loss -= random.uniform(0.02, 0.12)
    accuracy = min(0.99, 1 - loss / 2)
    wandb.log({"loss": loss, "accuracy": accuracy}, step=step)

run.finish()
