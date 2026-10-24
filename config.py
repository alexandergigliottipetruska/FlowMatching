from dataclasses import dataclass


# Create dataclass for configuration object
@dataclass
class TrainingConfig:
    batch_size: int
    learning_rate: float
    epochs: int
    seed: int
    device: str
    data_dir: str
    output_dir: str