import tyro
from pathlib import Path
from phosphobot.am import Gr00tTrainerConfig, Gr00tTrainer

config = tyro.cli(Gr00tTrainerConfig)

# Get the path to the cloned gr00t repo
path_to_gr00t_folder = Path(__file__).parent.parent.parent / "Isaac-GR00T"

# Useful parameters to change
config.training_params.num_epochs = 20
config.training_params.batch_size = 64
config.training_params.learning_rate = 0.0002

trainer = Gr00tTrainer(config)
trainer.train()

print("Training complete")
