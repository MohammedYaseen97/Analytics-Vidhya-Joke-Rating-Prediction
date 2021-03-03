import pandas as pd
import os
from sklearn import model_selection

import config

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(config.INPUT_PATH, 'train.csv'))
    
    #Shuffle the Dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    df["kfold"] = -1
    
    #Kfold Object
    kf = model_selection.KFold(n_splits=5)
    
    #Applying KFold
    for f, (t_, v_) in enumerate(kf.split(X=df)):
        df.loc[v_, "kfold"] = f
    
    #Shuffle the dataset again
    df = df.sample(frac=1).reset_index(drop=True)
    
    df.to_csv(os.path.join(config.INPUT_PATH, 'train_folds.csv'), index=False)