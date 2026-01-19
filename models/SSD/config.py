import torch

NUM_CLASSES = 4  # with_mask, without_mask, mask_incorrect
IMAGE_SIZE = 300

BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 1e-4

BACKBONE = "vgg" 
NEG_POS_RATIO = 3

# DEVICE = "cpu"
USE_GPU = True
DEVICE = torch.device("mps" if USE_GPU and torch.mps.is_available() else "cpu")

# Other training hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 2
