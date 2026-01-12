NUM_CLASSES = 3  # with_mask, without_mask, mask_incorrect
IMAGE_SIZE = 300

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3

BACKBONE = "vgg"  # "vgg" or "mobilenet"
NEG_POS_RATIO = 3

DEVICE = "cpu"