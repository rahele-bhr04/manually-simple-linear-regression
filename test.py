import numpy as np
import dataset as ds
import train


test_y_pred = np.array(list( (train.w * carat + train.b) for carat in ds.X_test)).flatten()


ss_res = np.sum((ds.y_test - test_y_pred) ** 2)
ss_tot = np.sum((ds.y_test - np.mean(ds.y_test)) ** 2)

r2 = 1 - (ss_res / ss_tot)