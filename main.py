from enum import unique
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
#from flask import Flask, request, make_response
from flask_sqlalchemy import SQLAlchemy
import flask_sqlalchemy
import pandas as pd
import pymysql
import sqlalchemy
from sklearn.preprocessing import MinMaxScaler
import pickle
import random
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Reshape, Dot, Flatten, concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import model_to_dot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import os

app = Flask(__name__)
app.secret_key = "Secret Key"

# API CONFIGURATION
USERNAME = "imamseptian"
PASSWORD = "imamseptian1234"
PUBLIC_IP_ADDRESS = "34.101.251.173"
DBNAME = "rating_db"
PROJECT_ID = "groovy-analyst-314808"
INSTANCE_NAME = "groovy-analyst-314808:asia-southeast2:ayohealthy"

app.config["SECRET_KEY"] = "xxxxxxx"
app.config["SQLALCHEMY_DATABASE_URI"] = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{PUBLIC_IP_ADDRESS}/{DBNAME}"

# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:''@localhost/rating_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db = SQLAlchemy(app)

# FOOD PHOTOS DIRECTORY
PHOTOS_FOLDER = os.path.join('static', 'photos')
app.config['UPLOAD_FOLDER'] = PHOTOS_FOLDER

class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), unique=True)
    name = db.Column(db.String(255))
    email = db.Column(db.String(255), unique=True)
    password = db.Column(db.String(255))

    def __init__(self, username, name, email, password):
        self.username = username
        self.name = name
        self.email = email
        self.password = password


class Foods(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    food_code = db.Column(db.Integer, unique=True)
    name = db.Column(db.String(255))
    category = db.Column(db.String(255))
    type = db.Column(db.String(255))
    calories = db.Column(db.Float)
    protein = db.Column(db.Float)
    carbs = db.Column(db.Float)
    fat = db.Column(db.Float)
    fiber = db.Column(db.Float)
    sugar = db.Column(db.Float)
    vitamin_a = db.Column(db.Float)
    vitamin_b6 = db.Column(db.Float)
    vitamin_b12 = db.Column(db.Float)
    vitamin_c = db.Column(db.Float)
    vitamin_d = db.Column(db.Float)
    vitamin_e = db.Column(db.Float)

    def __init__(self, food_code, name, category, type, calories, protein, carbs, fat, fiber, sugar, vitamin_a, vitamin_b6, vitamin_b12, vitamin_c, vitamin_d, vitamin_e):
        self.food_code = food_code
        self.name = name
        self.category = category
        self.type = type
        self.calories = calories
        self.protein = protein
        self.carbs = carbs
        self.fat = fat
        self.fiber = fiber
        self.sugar = sugar
        self.vitamin_a = vitamin_a
        self.vitamin_b6 = vitamin_b6
        self.vitamin_b12 = vitamin_b12
        self.vitamin_c = vitamin_c
        self.vitamin_d = vitamin_d
        self.vitamin_e = vitamin_e


class Ratings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    food_id = db.Column(db.Integer)
    food_code = db.Column(db.Integer)
    rating = db.Column(db.Integer)

    def __init__(self, user_id, food_id, food_code, rating):
        self.user_id = user_id
        self.food_id = food_id
        self.food_code = food_code
        self.rating = rating

class Food_images(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    food_id = db.Column(db.Integer)
    food_code = db.Column(db.Integer)
    image = db.Column(db.String(255))

    def __init__(self, food_id, food_code, image):
        self.food_id = food_id
        self.food_code = food_code
        self.image = image

# Load DF to API

predicted_df = pd.read_csv('RecommendationCSV/TF Recommendation Index 07-06-2021 13-25-36.csv') # DF that every food rating already predicted by TF model
predicted_df.rename(columns={'Unnamed: 0': 'user_id'}, inplace=True)
predicted_df = predicted_df.set_index('user_id')

food_df = pd.read_csv('ExportedDB/Exported Rated Food 07-06-2021 12-46-47.csv') # Food with nutrition data

all_code = list(food_df['food_code'].values)

# dict for every food_code to food name
food_name_dict = {}
for index, row in food_df.iterrows():
    if(row['food_code'] not in food_name_dict):
        food_name_dict[row['food_code']] = row['name']

knn_df = food_df.copy()
knn_df.drop(['food_id', 'food_code', 'name',
             'category', 'type'], axis=1, inplace=True)

# scale food_df
min_max_scaler = MinMaxScaler()
knn_df = min_max_scaler.fit_transform(knn_df)

# load KNN model
model = pickle.load(open('TrainedModel/Food KNN 07-06-2021 18-18-27.pkl', 'rb'))
distances, indices = model.kneighbors(knn_df)

# BASE URL VIEW , Not really used on this API
@app.route('/')
def Index():
    sql = sqlalchemy.text(
        'SELECT users.id,users.name,users.username, SUM(IF(ratings.id IS NULL, 0, 1)) AS rating_count FROM users LEFT JOIN ratings ON ratings.user_id = users.id GROUP BY 1')
    all_data = db.engine.execute(sql)

    return render_template("index.html", user_data=all_data)

# Recommendation based on DF that already predicted by TF model
# <id> ==> user_id (make sure user_id registered on DB)
@app.route('/recommendation/<id>')
def givetfrecommendation(id):
    current_user = Users.query.filter(Users.id == id).all()
    #check if user exist
    if(current_user):
        selected_user = current_user[0]
        user_json = {
            "id": selected_user.id,
            "username": selected_user.username,
            "name": selected_user.name,
            "email": selected_user.email
        }

        # find user data in predicted_df
        food_code_arr = []
        name_arr = []
        prediction_arr = []
        for key, value in predicted_df.loc[int(id)].items():
            food_code_arr.append(int(key))
            name_arr.append(food_name_dict[int(key)])
            prediction_arr.append(value)

        recommendation_columns = {'food_code': food_code_arr,
                                  'name': name_arr,
                                  'predicted_rating': prediction_arr,
                                  }
        # convert user food rating to its own dataframe
        rec_for_user = pd.DataFrame(data=recommendation_columns)
        rec_for_user = rec_for_user.sort_values(
            by='predicted_rating', ascending=False)
        
        # find food_code that already rated by the user
        user_rating = Ratings.query.filter(Ratings.user_id == id).all()
        rated_food = []
        for rate in user_rating:
            rated_food.append(rate.food_code)

        # exclude rated food from the recommendation
        non_rated_recommendation = rec_for_user.loc[(
            ~rec_for_user['food_code'].isin(rated_food))]

        recommended_food_code = []
        non_rated_recommendation = non_rated_recommendation.sort_values(
            by='predicted_rating', ascending=False)

        print(non_rated_recommendation.head(10))

        # get 10 recommended food_code
        for index, row in non_rated_recommendation.head(10).iterrows():
            recommended_food_code.append(row.food_code)

        food_data = db.session.query(Foods).filter(
            Foods.food_code.in_(recommended_food_code)).all()

        # get recommended foods nutrition detail
        arr_json = []
        content = {}
        shuffled_top10 = random.sample(food_data, len(food_data))

        for food in shuffled_top10:
            content = {'id': food.id, 'food_code': food.food_code, 'name': food.name, 'category': food.category,
                       'calories': food.calories, 'carbs': food.carbs, 'protein': food.protein, 'fat': food.fat, 'fiber': food.fiber, 'sugar': food.sugar,
                       'vitamin_a': food.vitamin_a, 'vitamin_b6': food.vitamin_b6, 'vitamin_b12': food.vitamin_b12, 'vitamin_c': food.vitamin_c, 'vitamin_d': food.vitamin_d,
                       'vitamin_e': food.vitamin_e,"image_url": "{}static/photos/{}.jpg".format(request.host_url,food.food_code)}
            arr_json.append(content)
            content = {}

        # return recommendation JSON response
        jsonku = {"status": "success",
                  "code": 200,
                  "message": "User Found",
                  "user_data": user_json,
                  "recommended_fnb": arr_json
                  }
        return jsonify(jsonku)
    # In case user not in DB    
    else:
        jsonku = {"status": "success",
                  "code": 404,
                  "message": "User Not Found"
                  }
        return jsonify(jsonku)

# Content based recommendation using scikit-learn KNN
# This DF works by finding the n most similar FNB based on the nutrition
# <id> ==> food_code (not food_id! check DB)
@app.route('/recommendation_fnb/<id>')
def itemrecommendation(id):
    user_food = Foods.query.filter(Foods.food_code == id).all()
    # make sure the food exist
    if(user_food):
        selected_food = user_food[0]
        current_food = {
            "id": selected_food.id,
            "food_code": selected_food.food_code,
            "name": selected_food.name,
            "category": selected_food.category,
            "type": selected_food.type,
            "calories": selected_food.calories,
            "carbs": selected_food.carbs,
            "fat": selected_food.fat,
            "sugar": selected_food.sugar,
            "fiber": selected_food.fiber,
            "vitamin_a": selected_food.vitamin_a,
            "vitamin_b6": selected_food.vitamin_b6,
            "vitamin_b12": selected_food.vitamin_b12,
            "vitamin_c": selected_food.vitamin_c,
            "vitamin_d": selected_food.vitamin_d,
            "vitamin_e": selected_food.vitamin_e,
            "image_url": "{}static/photos/{}.jpg".format(request.host_url,selected_food.food_code)
        }

        # get n most similar foods
        searched_food_code = all_code.index(int(id))
        searched_food_name = food_df.loc[food_df['food_code'] == int(id)].head(
            1).name.tolist()

        # n most similar food_code
        nearest_fnb = food_df.loc[indices[searched_food_code]
                                  ]['food_code'].tolist()

        food_data = db.session.query(Foods).filter(
            Foods.food_code.in_(nearest_fnb)).all()

        arr_json = []
        content = {}
        for food in food_data:

            # content = {'id':foods.id,'bruh':22}
            content = {'id': food.id, 'food_code': food.food_code, 'name': food.name, 'category': food.category,
                       'calories': food.calories, 'carbs': food.carbs, 'protein': food.protein, 'fat': food.fat, 'fiber': food.fiber, 'sugar': food.sugar,
                       'vitamin_a': food.vitamin_a, 'vitamin_b6': food.vitamin_b6, 'vitamin_b12': food.vitamin_b12, 'vitamin_c': food.vitamin_c, 'vitamin_d': food.vitamin_d,
                       'vitamin_e': food.vitamin_e, "image_url": "{}static/photos/{}.jpg".format(request.host_url,food.food_code)}
            arr_json.append(content)
            content = {}

        # return content-based recommendation JSON
        jsonku = {"status": "success",
                  "code": 200,
                  "message": "Item Found",
                  "detail_fnb": current_food,
                  "related_fnb": arr_json
                  }
        return jsonify(jsonku)
    #in case of user not found
    else:
        jsonku = {"status": "success",
                  "code": 404,
                  "message": "Item Not Found"
                  }
        return jsonify(jsonku)


# endpoint to search FNB by keyword
@app.route('/find_fnb', methods=['GET', 'POST'])
def find_fnb():
    if request.method == 'POST':
        request_data = request.get_json()
        sql = sqlalchemy.text(
            "SELECT * FROM foods WHERE name LIKE '%{}%';".format(request_data['keyword']))
        all_data = db.engine.execute(sql)
        
        arr_json = []
        content = {}
        num_food = 0
        for food in all_data:

            # content = {'id':foods.id,'bruh':22}
            content = {'id': food.id, 'food_code': food.food_code, 'name': food.name, 'category': food.category,
                       'calories': food.calories, 'carbs': food.carbs, 'protein': food.protein, 'fat': food.fat, 'fiber': food.fiber, 'sugar': food.sugar,
                       'vitamin_a': food.vitamin_a, 'vitamin_b6': food.vitamin_b6, 'vitamin_b12': food.vitamin_b12, 'vitamin_c': food.vitamin_c, 'vitamin_d': food.vitamin_d,
                       'vitamin_e': food.vitamin_e, "image_url": "{}static/photos/{}.jpg".format(request.host_url,food.food_code)}
            arr_json.append(content)
            content = {}
            num_food+=1
        print(num_food)
        if(num_food>0):
            jsonku = {"status": "success",
                    "code": 200,
                    "message": "Item Found",
                    "fnb_data": arr_json
                    }
            return jsonify(jsonku)
        else:
            jsonku = {"status": "success",
                    "code": 200,
                    "message": "Item Not Found",
                    "fnb_data": arr_json
                    }
            return jsonify(jsonku)

# TF Model
def RecommenderV2(n_users, n_food, n_dim):
    
    # User
    user = Input(shape=(1,))
    U = Embedding(n_users, n_dim)(user)
    U = Flatten()(U)
    
    # food
    food = Input(shape=(1,))
    M = Embedding(n_food, n_dim)(food)
    M = Flatten()(M)
    
    merged_vector = concatenate([U, M])
    dense_1 = Dense(128, activation='relu')(merged_vector)
    dense_2 = Dense(64, activation='relu')(dense_1)

    final = Dense(1)(dense_2)
    
    model = Model(inputs=[user, food], outputs=final)
    
    model.compile(optimizer=Adam(0.001),
                  loss='mean_squared_error')
    
    return model

# Endpoint to retrain the model and save the new TF & KNN Model
@app.route('/export_and_train')
def convert_food():
    # return "Hellow Flask Application"
    food_id = []
    food_code = []
    food_name = []
    food_category = []
    food_type = []
    food_calories = []
    food_protein = []
    food_carbs = []
    food_fat = []
    food_fiber = []
    food_sugar = []
    food_vitamin_a = []
    food_vitamin_b6 = []
    food_vitamin_b12 = []
    food_vitamin_c = []
    food_vitamin_d = []
    food_vitamin_e = []

    food_db = Foods.query.all()

    for food in food_db:
        food_id.append(food.id)
        food_code.append(food.food_code)
        food_name.append(food.name)
        food_category.append(food.category)
        food_type.append(food.type)
        food_calories.append(food.calories)
        food_protein.append(food.protein)
        food_carbs.append(food.carbs)
        food_fat.append(food.fat)
        food_fiber.append(food.fiber)
        food_sugar.append(food.sugar)
        food_vitamin_a.append(food.vitamin_a)
        food_vitamin_b6.append(food.vitamin_b6)
        food_vitamin_b12.append(food.vitamin_b12)
        food_vitamin_c.append(food.vitamin_c)
        food_vitamin_d.append(food.vitamin_d)
        food_vitamin_e.append(food.vitamin_e)

    food_colums = {'food_id': food_id,
                   'food_code': food_code,
                   'name': food_name,
                   'category': food_category,
                   'type': food_type,
                   'calories': food_calories,
                   'protein': food_protein,
                   'carbs': food_carbs,
                   'fat': food_fat,
                   'fiber': food_fiber,
                   'sugar': food_sugar,
                   'vitamin_a': food_vitamin_a,
                   'vitamin_b6': food_vitamin_b6,
                   'vitamin_b12': food_vitamin_b12,
                   'vitamin_c': food_vitamin_c,
                   'vitamin_d': food_vitamin_d,
                   'vitamin_e': food_vitamin_e
                   }
    food_df = pd.DataFrame(data=food_colums)
    food_csv_title = 'ExportedDB/Exported Food {}.csv'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    food_df.to_csv(food_csv_title, index=False)

    user_id = []
    user_name = []
    user_username = []
    user_email = []

    user_db = Users.query.all()

    for user in user_db:
        user_id.append(user.id)
        user_name.append(user.name)
        user_username.append(user.username)
        user_email.append(user.email)

    user_columns = {'user_id': user_id,
                    'name': user_name,
                    'username': user_username,
                    'email': user_email,
                    }
    user_df = pd.DataFrame(data=user_columns)
    user_csv_title = 'ExportedDB/Exported User {}.csv'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    user_df.to_csv(user_csv_title, index=False)

    rating_id = []
    rating_user_id = []
    rating_food_id = []
    rating_food_code = []
    rating_rating = []

    rating_db = Ratings.query.all()

    for rate in rating_db:
        rating_id.append(rate.id)
        rating_user_id.append(rate.user_id)
        rating_food_id.append(rate.food_id)
        rating_food_code.append(rate.food_code)
        rating_rating.append(rate.rating)

    rating_columns = {'rating_id': rating_id,
                      'user_id': rating_user_id,
                      'food_id': rating_food_id,
                      'food_code': rating_food_code,
                      'rating': rating_rating,
                      }
    rating_df = pd.DataFrame(data=rating_columns)
    rating_csv_title = 'ExportedDB/Exported Rating {}.csv'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    rating_df.to_csv(rating_csv_title, index=False)

    exported_food_df = pd.read_csv(food_csv_title)
    exported_rating_df = pd.read_csv(rating_csv_title)

    rated_food = exported_rating_df['food_code'].unique().tolist()
    rated_df = exported_food_df.loc[(exported_food_df['food_code'].isin(rated_food))]
    rated_food_title = 'ExportedDB/Exported Rated Food {}.csv'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    rated_df.to_csv(rated_food_title, index=False)

    train_food_df = pd.read_csv(rated_food_title)
    train_rating_df = pd.read_csv(rating_csv_title)

    user_enc = LabelEncoder()
    train_rating_df['User_ID'] = user_enc.fit_transform(train_rating_df['user_id'])

    food_enc = LabelEncoder()
    train_rating_df['Food_ID'] = food_enc.fit_transform(train_rating_df['food_code'])

    userid_nunique = train_rating_df['User_ID'].nunique()
    food_unique = train_rating_df['Food_ID'].nunique()

    print('Using tensorflow version:', tf.__version__)
   
    ori_df = pd.read_csv(rated_food_title)
    knn_df = ori_df.copy()
    knn_df.drop(['food_id','food_code','name','category','type'],axis =1 ,inplace = True)
    
    
    min_max_scaler = MinMaxScaler()
    # knn_df=df
    knn_df = min_max_scaler.fit_transform(knn_df)
    nbrs = NearestNeighbors(n_neighbors=10, algorithm='ball_tree').fit(knn_df)
    knn_title = 'TrainedModel/Food KNN {}.pkl'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    pickle.dump(nbrs, open(knn_title, 'wb'))


    return "DONE"


if __name__ == "__main__":
    app.run(debug=True)
