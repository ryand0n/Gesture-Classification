from get_data import collect_data
import model.build_model as bm
import pandas as pd

raised = "data/raised_hand_data.csv"
clap = "data/clap_data_1.csv"

# make sure you have the correct port in arduino in get_data.py, just for collecting clap data
# can comment this out after clap data has been collected
#collect_data(25, clap)

#raised_df = pd.read_csv(raised) 
clap_df = pd.read_csv(clap)

#raised_df = bm.transform(raised_df, "raise")
#clap_df = bm.transform(clap_df, "clap")

#df = bm.connect_df(raised_df, clap_df)
#model = bm.fit(df)

# predict gestures on 10 gestures
# uncomment when model has been trained
#collect_data(10, None, predicter=True, model=model)
