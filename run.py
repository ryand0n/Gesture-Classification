#from get_data import collect_data
import model.build_model as bm
import pandas as pd
import numpy as np

raised = "data/raised_hand_data.csv"
clap = "data/clap_data.csv"
clap_1 = "data/clap_data_1.csv"

# make sure you have the correct port in arduino in get_data.py, just for collecting clap data
# can comment this out after clap data has been collected
#collect_data(25, clap)

raised_df = pd.read_csv(raised) 
clap_df = pd.read_csv(clap)
clap_1_df = pd.read_csv(clap_1)

raised_df = bm.transform(raised_df, "raise")
clap_df = bm.transform(clap_df, "clap")
clap_1_df = bm.transform(clap_1_df, "clap")

df = bm.connect_df(raised_df, clap_df)
df = bm.connect_df(df, clap_1_df)
model = bm.fit(df)

def acc(df):
    pred = []
    for i in range(len(df)):
        row = df.loc[i][:-1].to_numpy().reshape(1, -1)
        pred.append(model.predict(row)[0])

    return (pred == np.zeros(len(df))).mean()

best_feat = ['z_mean', 'x_min', 'y_min', 'z_min', 'z_slope', 'z_std',
       'x_seg_slope_4', 'y_seg_slope_4', 'z_seg_slope_3', 'z_seg_slope_4',
       'target']

print(acc(clap_1_df[best_feat]))

# for i in range(len(raised_df)):
#     row = raised_df.loc[i][:-1].to_numpy().reshape(1, -1)
#     print(model.predict(row)[0])

# predict gestures on 10 gestures
# uncomment when model has been trained
#collect_data(10, None, predicter=True, model=model)
