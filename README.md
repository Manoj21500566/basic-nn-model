# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Neural network regression is a supervised learning method, and therefore requires a tagged dataset, which includes a label column. Because a regression model predicts a numerical value, the label column must be a numerical data type. You can train the model by providing the model and the tagged dataset as an input to Train Model.

In this experiment we need to develop a Neural Network Regression Model so first we need to create a dataset (i.e : an excel file with some inputs as well as corresponding outputs).Then upload the sheet to drive then using corresponding code open the sheet and then import the required python libraries for porcessing.

Use df.head to get the first 5 values from the dataset or sheet.Then assign x and y values for input and coressponding outputs.Then split the dataset into testing and training,fit the training set and for the model use the "relu" activation function for the hidden layers of the neural network (here two hidden layers of 4 and 6 neurons are taken to process).To check the loss mean square error is uesd.Then the testing set is taken and fitted, at last the model is checked for accuracy via preiction.

## Neural Network Model

![neural-network-model](https://user-images.githubusercontent.com/94588708/187125921-ae2e0e53-38da-44e2-8658-1bcb91242d84.png)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

Developed by : Anto Richard.S
Reg.no : 21222124005
Program to develop a neural network regression model..
````
### To Read CSV file from Google Drive :

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

### Authenticate User:

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

### Open the Google Sheet and convert into DataFrame :

worksheet = gc.open('data').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns = rows[0])
df = df.astype({'Input':'int','Output':'int'})

### Import the packages :

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df.head()

X = df[['Input']].values
y = df[['Output']].values
X

### Split Training and testing set :

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 42)

### Pre-processing the data :

Scaler = MinMaxScaler()
Scaler.fit(X_train)
Scaler.fit(X_test)

X_train1 = Scaler.transform(X_train)
X_test1 = Scaler.transform(X_test)

X_train1

### Model :

ai_brain = Sequential([
    Dense(4,activation = 'relu'),
    Dense(6,activation = 'relu'),
    Dense(1)
])

ai_brain.compile(
    optimizer = 'rmsprop',
    loss = 'mse'
)

ai_brain.fit(X_train1,y_train,epochs = 4000)

### Loss plot :

loss_df = pd.DataFrame(ai_brain.history.history)

loss_df.plot()

### Testing with the test data and predicting the output :

ai_brain.evaluate(X_test1,y_test)

X_n1 = [[89]]

X_n1_1 = Scaler.transform(X_n1)

ai_brain.predict(X_n1

## Dataset Information

![out1](https://user-images.githubusercontent.com/94588708/187126189-6bf87d2a-5869-4ca8-8935-b4896492f407.png)

```
## OUTPUT

### Initiation of program :

![out2](https://user-images.githubusercontent.com/94588708/187126561-979112c7-6fbe-464d-a797-61d9abf15b4e.png)



![out3](https://user-images.githubusercontent.com/94588708/187127243-ec263ae3-776b-4c3a-a628-697e4264c77a.png)



![out4](https://user-images.githubusercontent.com/94588708/187127259-6842a127-5dee-45d7-8f4f-299491405d6f.png)




### Training Loss Vs Iteration Plot :

![out5](https://user-images.githubusercontent.com/94588708/187127562-a40be418-09c8-4899-8c45-28111b75d53b.png)



![out6](https://user-images.githubusercontent.com/94588708/187127636-69f888e5-deba-43cd-9322-ae9a72aeb8c2.png)

### Test Data Root Mean Squared Error :

![out7](https://user-images.githubusercontent.com/94588708/187127956-4587cd78-fba5-44c4-8637-b57c8201d32b.png)




### New Sample Data Prediction

![out8](https://user-images.githubusercontent.com/94588708/187128018-594d4927-325e-4fe7-823a-d14ec2fe56b6.png)


## RESULT

Thus a neural network model for regression using the given dataset is written and executed successfully.


