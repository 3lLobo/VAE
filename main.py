import tensorflow as tf
import numpy as np
import time
from IPython import display
from models.CVAE import VanillaCVAE
from max_pooling import MPGM


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Num TPUs Available: ", len(tf.config.experimental.list_physical_devices('TPU')))


if __name__ == "__main__":

"""
    At some point in time I will move the code from GVAE.py to here and make the text code from max_pooling.py and actual test.
"""
