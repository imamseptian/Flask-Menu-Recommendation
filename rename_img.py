import os
import pandas as pd

food_df = pd.read_csv('Exported Rated Food 07-06-2021 16-35-04.csv')


# remove end number
# for filename in os.listdir(os.path.join(os.getcwd(), 'static/photos/')):
#     # print(filename)
#     file_name=os.path.splitext(os.path.join(os.getcwd(), 'static/photos/')+filename)[0]
#     file_extension=os.path.splitext(os.path.join(os.getcwd(), 'static/photos/')+filename)[1]
#     # print(file_extension)
#     old_file = file_name+file_extension
#     new_file = file_name[:-1]+file_extension
#     # print(filename)
#     # print(file_extension)
#     # break

#     os.rename(old_file, new_file)
    

# food_not_exist=[]
for index, row in food_df.iterrows():
    if (os.path.isfile(os.path.join(os.getcwd(), 'static/photos/{}.jpg'.format(row['name'])))):
        file_name=os.path.splitext(os.path.join(os.getcwd(), 'static/photos/{}.jpg'.format(row['name'])))[0]
        file_extension=os.path.splitext(os.path.join(os.getcwd(), 'static/photos/{}.jpg'.format(row['name'])))[1]
        
        old_file = file_name+file_extension
 

        new_file = (os.path.join(os.getcwd(), 'static/photos/{}.jpg'.format(row['food_code'])))
        os.rename(old_file, new_file)
