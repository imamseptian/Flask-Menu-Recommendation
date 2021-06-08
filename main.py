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
# from tensorflow.keras.layers import Input, Embedding, Reshape, Dot, Flatten, concatenate, Dense, Dropout
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

# Google Cloud SQL (change this accordingly)
# USERNAME = "imamseptian"
# PASSWORD = "imamseptian1234"
# PUBLIC_IP_ADDRESS = "34.126.174.105"
# DBNAME = "rating_db"
# PROJECT_ID = "groovy-analyst-314808"
# INSTANCE_NAME = "groovy-analyst-314808:asia-southeast1:collectrating"

USERNAME = "imamseptian"
PASSWORD = "imamseptian1234"
PUBLIC_IP_ADDRESS = "34.101.251.173"
DBNAME = "rating_db"
PROJECT_ID = "groovy-analyst-314808"
INSTANCE_NAME = "groovy-analyst-314808:asia-southeast2:ayohealthy"


# configuration
app.config["SECRET_KEY"] = "xxxxxxx"
app.config["SQLALCHEMY_DATABASE_URI"] = f"mysql+pymysql://{USERNAME}:{PASSWORD}@{PUBLIC_IP_ADDRESS}/{DBNAME}"

# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:''@localhost/rating_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db = SQLAlchemy(app)
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


food_name_dict = {}
# predicted_df = pd.read_csv('TF Recommendation Index.csv')
predicted_df = pd.read_csv('RecommendationCSV/TF Recommendation Index 07-06-2021 13-25-36.csv')
# food_df = pd.read_csv('Rated Food.csv')
food_df = pd.read_csv('ExportedDB/Exported Rated Food 07-06-2021 12-46-47.csv')
predicted_df.rename(columns={'Unnamed: 0': 'user_id'}, inplace=True)
predicted_df = predicted_df.set_index('user_id')
all_code = list(food_df['food_code'].values)

for index, row in food_df.iterrows():
    if(row['food_code'] not in food_name_dict):
        food_name_dict[row['food_code']] = row['name']

knn_df = food_df.copy()
knn_df.drop(['food_id', 'food_code', 'name',
             'category', 'type'], axis=1, inplace=True)

min_max_scaler = MinMaxScaler()
# knn_df=df
knn_df = min_max_scaler.fit_transform(knn_df)
model = pickle.load(open('TrainedModel/Food KNN 07-06-2021 18-18-27.pkl', 'rb'))
distances, indices = model.kneighbors(knn_df)


@app.route('/')
def Index():
    # return "Hellow Flask Application"
    # all_data = Users.query.all()
    sql = sqlalchemy.text(
        'SELECT users.id,users.name,users.username, SUM(IF(ratings.id IS NULL, 0, 1)) AS rating_count FROM users LEFT JOIN ratings ON ratings.user_id = users.id GROUP BY 1')
    all_data = db.engine.execute(sql)

    return render_template("index.html", user_data=all_data)


@app.route('/recommendation/<id>')
def givetfrecommendation(id):
    current_user = Users.query.filter(Users.id == id).all()
    if(current_user):
        selected_user = current_user[0]
        user_json = {
            "id": selected_user.id,
            "username": selected_user.username,
            "name": selected_user.name,
            "email": selected_user.email
        }

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

        rec_for_user = pd.DataFrame(data=recommendation_columns)
        rec_for_user = rec_for_user.sort_values(
            by='predicted_rating', ascending=False)
        # food_data = db.session.query(Foods).filter(Foods.id.notin_(rated_food))
        user_rating = Ratings.query.filter(Ratings.user_id == id).all()
        rated_food = []
        for rate in user_rating:
            rated_food.append(rate.food_code)
        # print(rated_food)
        non_rated_recommendation = rec_for_user.loc[(
            ~rec_for_user['food_code'].isin(rated_food))]
        recommended_food_code = []
        non_rated_recommendation = non_rated_recommendation.sort_values(
            by='predicted_rating', ascending=False)
        print(non_rated_recommendation.head(10))
        for index, row in non_rated_recommendation.head(10).iterrows():
            recommended_food_code.append(row.food_code)

        food_data = db.session.query(Foods).filter(
            Foods.food_code.in_(recommended_food_code)).all()

        arr_json = []
        content = {}
        # shuffled_top10 = random.sample(food_data, len(food_data))
        shuffled_top10 = food_data
        for food in shuffled_top10:

            # content = {'id':foods.id,'bruh':22}
            content = {'id': food.id, 'food_code': food.food_code, 'name': food.name, 'category': food.category,
                       'calories': food.calories, 'carbs': food.carbs, 'protein': food.protein, 'fat': food.fat, 'fiber': food.fiber, 'sugar': food.sugar,
                       'vitamin_a': food.vitamin_a, 'vitamin_b6': food.vitamin_b6, 'vitamin_b12': food.vitamin_b12, 'vitamin_c': food.vitamin_c, 'vitamin_d': food.vitamin_d,
                       'vitamin_e': food.vitamin_e,"image_url": "{}static/photos/{}.jpg".format(request.host_url,food.food_code)}
            arr_json.append(content)
            content = {}

        jsonku = {"status": "success",
                  "code": 200,
                  "message": "User Found",
                  "user_data": user_json,
                  "recommended_food": arr_json
                  }
        return jsonify(jsonku)
    else:
        jsonku = {"status": "success",
                  "code": 404,
                  "message": "User Not Found"
                  }
        return jsonify(jsonku)


@app.route('/recommendation_fnb/<id>')
def itemrecommendation(id):
    user_food = Foods.query.filter(Foods.food_code == id).all()
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

        searched_food_code = all_code.index(int(id))
        searched_food_name = food_df.loc[food_df['food_code'] == int(id)].head(
            1).name.tolist()
        print(searched_food_name)
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

        jsonku = {"status": "success",
                  "code": 200,
                  "message": "Item Found",
                  "food_data": current_food,
                  "related_food": arr_json
                  }
        return jsonify(jsonku)
    else:
        jsonku = {"status": "success",
                  "code": 404,
                  "message": "Item Not Found"
                  }
        return jsonify(jsonku)


@app.route('/testjson/<id>')
def myjson(id):
    user_food = Foods.query.filter(Foods.food_code == id).all()
    # user_food = Ratings.query.filter(Ratings.food_code == id).all()
    if(user_food):
        selected_food = user_food[0]
        current_food = {
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
        }

        jsonku = {"status": "success",
                  "code": 200,
                  "message": "Item Found",
                  "food_data": current_food,
                  "related_food": []
                  }
        return jsonify(jsonku)
    else:
        jsonku = {"status": "success",
                  "code": 404,
                  "message": "Item Not Found"
                  }
        return jsonify(jsonku)


@app.route('/find_fnb', methods=['GET', 'POST'])
def find_fnb():
    if request.method == 'POST':
        request_data = request.get_json()
        sql = sqlalchemy.text(
            "SELECT * FROM foods WHERE name LIKE '%{}%';".format(request_data['keyword']))
        all_data = db.engine.execute(sql)

        arr_json = []
        content = {}
        for food in all_data:

            # content = {'id':foods.id,'bruh':22}
            content = {'id': food.id, 'food_code': food.food_code, 'name': food.name, 'category': food.category,
                       'calories': food.calories, 'carbs': food.carbs, 'protein': food.protein, 'fat': food.fat, 'fiber': food.fiber, 'sugar': food.sugar,
                       'vitamin_a': food.vitamin_a, 'vitamin_b6': food.vitamin_b6, 'vitamin_b12': food.vitamin_b12, 'vitamin_c': food.vitamin_c, 'vitamin_d': food.vitamin_d,
                       'vitamin_e': food.vitamin_e, "image_url": "{}static/photos/{}.jpg".format(request.host_url,food.food_code)}
            arr_json.append(content)
            content = {}
        print(arr_json)
        jsonku = {"status": "success",
                  "code": 200,
                  "message": "Item Found",
                  "food_data": arr_json
                  }
        return jsonify(jsonku)

def RecommenderV2(n_users, n_food, n_dim):
    
    # User
    user = Input(shape=(1,))
    U = Embedding(n_users, n_dim)(user)
    U = Flatten()(U)
    
    # food
    movie = Input(shape=(1,))
    M = Embedding(n_food, n_dim)(movie)
    M = Flatten()(M)
    
    # Gabungkan disini
    merged_vector = concatenate([U, M])
    dense_1 = Dense(128, activation='relu')(merged_vector)
    dense_2 = Dense(64, activation='relu')(dense_1)
    # dropout = Dropout(0.5)(dense_1)
    final = Dense(1)(dense_2)
    
    model = Model(inputs=[user, movie], outputs=final)
    
    model.compile(optimizer=Adam(0.001),
                  loss='mean_squared_error')
    
    return model

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


    # best_state_val_loss = 0
    # best_state_end_val_loss = 0
    # lowest_val_loss = 999999
    # lowest_end_val_loss = 999999
    # for i in range(1,100):
    #     print("------------------ ITERATION NUMBER {} ------------------".format(i))
    #     model = RecommenderV2(userid_nunique, food_unique, 32)
    #     X = train_rating_df.drop(['rating'], axis=1)
    #     y = train_rating_df['rating']

    #     X_train, X_val, y_train, y_val = train_test_split(X, y,
    #                                                     test_size=.2,
    #                                                     stratify=y,
    #                                                     random_state=i)
    #     # X_train, X_val, y_train, y_val = train_test_split(X, y,
    #     #                                                 test_size=.2)

    #     model_title = 'TrainedModel/Model {}.h5'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    #     checkpoint = ModelCheckpoint(model_title, monitor='val_loss', verbose=0, save_best_only=True)
    #     val_loss_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    #     print(model_title)
    #     history = model.fit(x=[X_train['User_ID'], X_train['Food_ID']],
    #                     y=y_train,
    #                     batch_size=64,
    #                     epochs=100,
    #                     verbose=1,
    #                     validation_data=([X_val['User_ID'], X_val['Food_ID']], y_val),
    #                     callbacks=[val_loss_cb,checkpoint])

    #     training_loss2 = history.history['loss']
    #     test_loss2 = history.history['val_loss']
    #     i_min_val_loss = min(test_loss2)
    #     i_end_val_loss = test_loss2[len(test_loss2)-1]
    #     if(i_min_val_loss<lowest_val_loss):
    #         best_state_val_loss = i
    #         lowest_val_loss = i_min_val_loss

    #     if(i_end_val_loss<lowest_end_val_loss):
    #         best_state_end_val_loss = i
    #         lowest_end_val_loss = i_end_val_loss
    #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')         

    # print("Lower val_loss : {} on random_state : {}".format(lowest_val_loss,best_state_val_loss))
    # print("Lower end val_loss : {} on random_state : {}".format(lowest_end_val_loss,best_state_end_val_loss))

    # model = RecommenderV2(userid_nunique, food_unique, 32)
    # X = train_rating_df.drop(['rating'], axis=1)
    # y = train_rating_df['rating']

    # X_train, X_val, y_train, y_val = train_test_split(X, y,
    #                                                 test_size=.2,
    #                                                 stratify=y,
    #                                                 random_state=87)
    # # X_train, X_val, y_train, y_val = train_test_split(X, y,
    # #                                                 test_size=.2)

    # model_title = 'TrainedModel/Model {}.h5'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    # checkpoint = ModelCheckpoint(model_title, monitor='val_loss', verbose=0, save_best_only=True)
    # val_loss_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    # history = model.fit(x=[X_train['User_ID'], X_train['Food_ID']],
    #                 y=y_train,
    #                 batch_size=64,
    #                 epochs=100,
    #                 verbose=1,
    #                 validation_data=([X_val['User_ID'], X_val['Food_ID']], y_val),
    #                 callbacks=[val_loss_cb,checkpoint])


    # train_rating_df_title = 'TrainedModel/train_rating_df {}.csv'.format(datetime.now().strftime("%d-%m-%Y %H-%M-%S"))
    # train_rating_df.to_csv(train_rating_df_title, index=False)
   
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




# @app.route('/delete_non_rated')
# def deletenorate():
#     # all_food = pd.read_csv('Exported Food.csv')
#     all_food = pd.read_csv('Exported Food 07-06-2021 16-35-04.csv')
#     # rated_food_only = pd.read_csv('Exported Rating.csv')
#     rated_food_only = pd.read_csv('Exported Rating 07-06-2021 16-35-04.csv')
#     rated_index_only = rated_food_only['food_code'].unique().tolist()
#     food_code_not_rated = all_food.loc[(
#         ~all_food['food_code'].isin(rated_index_only)), 'food_code'].tolist()
#     # not_rate_fnb = db.session.query(Foods).filter(Foods.food_code.in_(food_code_not_rated)).all()
#     delete_q = Foods.__table__.delete().where(
#         Foods.food_code.in_(food_code_not_rated))
#     db.session.execute(delete_q)
#     db.session.commit()

#     return "SUCCESS"


@app.route('/nowtime')
def testnow():
    # PEOPLE_FOLDER = os.path.join(os.getcwd(), 'photos')

    

    # return url_for('static',filename='photos/Almond milk, sweetened, chocolate2.jpg')
    # return render_template("cobaimg.html")
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Almond milk, sweetened, chocolate2.jpg')
    # return render_template("cobaimg.html", user_image = full_filename)
    # return url_for('static',filename='photos/Almond milk, sweetened, chocolate2.jpg')
    print(request.host_url)
    jsonku = {"status": "success",
                  "code": 200,
                  "message": "User Found",
                  "url": "{}static/photos/Almond milk, sweetened, chocolate2.jpg".format(request.host_url)
            
                  }
    return jsonify(jsonku)

# @app.route('/insert_image')
# def imginsert():
#     all_food = Foods.query.all()
#     for food in all_food:

#         # print(str(food.food_code)+'.jpg')
#         my_data = Food_images(food.id,food.food_code,str(food.food_code)+'.jpg')
#         db.session.add(my_data)
#         db.session.commit()

#     return 'SUCCESS'



if __name__ == "__main__":
    app.run(debug=True)
