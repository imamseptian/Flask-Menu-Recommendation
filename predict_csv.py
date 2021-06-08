# THIS CODE IS FOR CREATING PREDICTED RATING DATAFRAME USING TRAINED TF MODEL

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from datetime import datetime

# Load Rating DF, Food DF, and trained tf model
rating_csv_title = 'TrainedModel/train_rating_df 07-06-2021 16-35-09.csv'
rating = pd.read_csv(rating_csv_title)

model_title = 'TrainedModel/Model 07-06-2021 16-35-05.h5'
model = load_model(model_title)

food_csv_title = 'ExportedDB/Exported Rated Food 07-06-2021 16-35-04.csv'
food_df = pd.read_csv(food_csv_title)

print('Creating Zeros DF')
zeros_df = pd.DataFrame(0,index=sorted(rating['user_id'].unique()),columns=food_df['food_code'].tolist())

# creating food and user dict
user_dict = {}
food_dict = {}
for index, row in rating.iterrows():
  if(row['user_id'] not in user_dict):
    user_dict[row['user_id']] = row['User_ID']

  if(row['food_code'] not in food_dict):
    food_dict[row['food_code']] = row['Food_ID']

print('Start creating recommendation CSV')
total_user = len(zeros_df)
step_loop =0
for index, row in zeros_df.iterrows():
  arr_rated = rating.loc[rating['user_id']==index,'food_code'].tolist()
  for rate in arr_rated:
    zeros_df.loc[index,rate]=rating.loc[(rating['user_id']==index)&(rating['food_code']==rate),'rating'].tolist()[0]

  not_rated = food_df.loc[(~food_df['food_code'].isin(arr_rated)),'food_code'].tolist()
  for rate in not_rated:
    try:

      zeros_df.loc[index,rate]=model.predict([np.array([user_dict[index]]), np.array([food_dict[rate]])])[0][0]
    except:
      zeros_df.loc[index,rate]=0
    # zeros_df.loc[index,rate]=69
  step_loop+=1
  print('Filing Row {} / {}'.format(step_loop,total_user))

# saved predicted df
tf_rec_csv_title = 'RecommendationCSV/TF Recommendation Index {}.csv'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
zeros_df.to_csv(tf_rec_csv_title, index=True)