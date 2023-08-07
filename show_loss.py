import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

LOSS_PATH='model/loss.npy'


loss = np.load(LOSS_PATH)
print(loss.shape)
print(loss)

plt.plot(loss)
plt.show()


