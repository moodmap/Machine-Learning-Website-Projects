'''
Creating a very basic linear regression prediction model

The data will be fake house prices based on number of rooms, age and volume of swimming pool

=== A ===

1) Create the fake data  || == TICK == ||

2) Turn this into a CSV || == TICK == ||

3) Train a basic model on this data, splitting the CSV into train and test data || == TICK == ||

=== B ===

1) Can load the model || == TICK == ||

2) Model can output a prediction based on user input || == TICK == ||

=== C ===

1) Create a basic Django local webserver || == TICK == ||

2) ..with a home page that user can type in information, click a button and this information
appears below that || == Semi TICK - appears on a different page (hoping to improve) == ||

3) Create a basic python script that I/O the same || == TICK == ||

4) Replace this script with A3 script || == TICK == ||

=== D ===

1) Prepare an AWS instance (python3, django, pip - pandas, etc) || == TICK == ||

2) Get B working here || == TICK == ||

3) Configure the DNS settings to connect to a specific url || == TICK == ||

=== E ===

1) Share on Linkedin, describing the importance of the process

2) Upload to my gitlab

3) Add to CV

4) Plan the next project (maybe image classification)

'''


'''
A1 - creating fake data
'''

import pandas as pd
import sklearn
import numpy as np
from random import randint

half_num_houses_data = 100000


'''
Creating the ids 1 to [number]
'''
def create_id(starting_id, num):
    id_list = []
    x = starting_id
    for i in range (num):
        id_list.append(x)
        x+=1
    return id_list

'''
Creating the random number of rooms, creating parameters to allow diff between cheap and expensive
'''

def create_data(min, max, num):
    
    data_list = []
    for i in range (num):
        data_list.append(randint(min, max))
    return data_list

cheap_house_id = create_id(1, half_num_houses_data)
cheap_house_numRooms = create_data(1, 3, half_num_houses_data)
cheap_house_age = create_data(5, 100, half_num_houses_data)
cheap_house_poolVolume = create_data(8, 30, half_num_houses_data)
cheap_house_price = create_data(10000,200000,half_num_houses_data)

expensive_house_id = create_id((half_num_houses_data+1), half_num_houses_data)
expensive_house_numRooms = create_data(3, 12, half_num_houses_data)
expensive_house_age = create_data(5, 100, half_num_houses_data)
expensive_house_poolVolume = create_data(15, 80, half_num_houses_data)
expensive_house_price = create_data(200000,1000000,half_num_houses_data)

df = pd.DataFrame(data={
    "id":cheap_house_id+expensive_house_id,
    "numRooms":cheap_house_numRooms+expensive_house_numRooms,
    "age":cheap_house_age+expensive_house_age,
    "poolVolume":cheap_house_poolVolume+expensive_house_poolVolume,
    "housePrice":cheap_house_price+expensive_house_price
})

df.to_csv("fake_house_data"+".csv", sep=',',index=False)


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
df = pd.read_csv('fake_house_data.csv')



'''
Preparing the data
'''
X = df.drop(['housePrice','id'], axis=1)
y = df['housePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)

with open("house_price_model.pkl", 'wb') as file:  
    pickle.dump(model, file)
    
    
    
import pickle

with open("house_price_model.pkl", 'rb') as file:  
    model = pickle.load(file)
    
    
user_numRooms = input("How many rooms?: ")
user_age = input("How old is the house (years)?: ")
user_poolVol = input ("What is the volume of your swimming pool (Metres cubed)?: ")

if user_numRooms.isdigit()==False or user_age.isdigit()==False or user_poolVol.isdigit()==False:
    print('You didnt enter numbers. Try again')
else:
    prediction = model.predict([[int(user_numRooms), int(user_age), int(user_poolVol)]])
    print("We predict that your house is worth: £{:,}.00".format(int(prediction)))
    if prediction > 1000000:
        print("Wow you have a big expensive house. Is so good.")
