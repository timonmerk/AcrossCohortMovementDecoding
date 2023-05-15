import pandas as pd
import numpy as np

coords = np.array([[36.5, -67.5, 54], [39.5, -57.5, 58.5], [41, -46.5, 63], [41, -35.5, 65], [40.5, -24.5, 65.5], [39.5, -14, 64]])

df =pd.DataFrame(coords, columns=["x", "y", "z"])
df.to_csv("electrodes_new_patient.csv")