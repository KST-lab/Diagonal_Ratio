import os
import numpy as np
import pandas as pd

for VIEW in ['AP', 'LA']:
    for MODEL_NAME in ['ResNet50V2', 'ResNet101V2', 'ResNet152V2']:
        CSV_PATH = os.path.join('csv', 'ResNet', VIEW + '_' + MODEL_NAME + '.csv')
        df = pd.read_csv(CSV_PATH,index_col=0)
        df = df.drop(columns=['0'])
        print(VIEW, MODEL_NAME, np.mean(list(df.mean())))
        # print(df.mean())
        # quit()
