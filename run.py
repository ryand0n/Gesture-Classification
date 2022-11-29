from get_data import collect_data
import model.build_model as bm
import pandas as pd

raised = "data/raised_hand_data.csv"
clap = "data/clap_data.csv"

# make sure you have the correct port in arduino in get_data.py, just for collecting clap data
collect_data(25, clap)

raised_df = pd.DataFrame(raised) 
clap_df = pd.DataFrame(clap)
df = bm.connect_df(raised_df, clap_df)
df = bm.transform(df)
model = bm.fit(df)

# predict gestures on 10 gestures
#collect_data(10, None, predicter=True, model=model)
