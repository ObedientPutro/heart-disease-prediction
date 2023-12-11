import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Dataset Load
dataset = pd.read_csv('heart.csv')

# String
la=str()
scoreRate=str()

# Prediction Function
class Predictor:

    def has_disease(self, row):
        self.train(self)
        return True if self.predict(self, row) == 1 else False

    @staticmethod
    def train(self):
        self.standardScaler = StandardScaler()
        columns_to_scale = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        dataset[columns_to_scale] = self.standardScaler.fit_transform(dataset[columns_to_scale])
        y = dataset['target']
        X = dataset.drop(['target'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=8)
        self.knn_classifier.fit(X, y)
        score = self.knn_classifier.score(X_test, y_test)
        global scoreRate
        scoreRate = str(score)
        print('--Training Complete--')
        print('Score: ' + str(score))

    @staticmethod
    def predict(self, row):
        user_df = np.array(row).reshape(1, 13)
        user_df = self.standardScaler.transform(user_df)
        predicted = self.knn_classifier.predict(user_df)
        print("Predicted: " + str(predicted[0]))
        return predicted[0]

# Onclick Function Button
def startPredict():
    row=[[age,sex,cp,tbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]
    print(row)
    predictor = Predictor()
    o = predictor.has_disease(row)
    st.subheader("Prediction Result")
    st.write("Accuracy: " + scoreRate)
    if (o == True):
        print("Pasien Terdiagnosa Mengalami Penyakit Jantung")
        la="Pasien Terdiagnosa Mengalami Penyakit Jantung"
        st.write(la)
    else:
        print("Pasien Sehat")
        la="Pasien Sehat"
        st.write(la)
    return True

# Input Field
st.title('Heart Disease Predictor')
st.subheader('Fill your Details')

age = st.number_input('How old are you?')

sex = st.radio(
    "Choose a gender",
    [0, 1],
    captions = ['Female', 'Male']
)

cp = st.radio(
    "cp: chest pain type (0-3)",
    [0, 1, 2, 3]
)

tbps = st.number_input('trestbps: resting blood pressure (ex: 145)')

chol = st.number_input('chol: serum cholestoral in mg/dl (ex: 233)')

fbs = st.radio(
    "fbs: (fasting blood sugar > 120 mg/dl)",
    [0, 1],
    captions = ['False', 'True']
)

restecg = st.radio(
    "restecg: resting electrocardiographic results",
    [0, 1, 2]
)

thalach = st.number_input('thalach: maximum heart rate achieved (ex: 187)')

exang = st.radio(
    "exang: exercise induced angina",
    [0, 1],
    captions = ['No', 'Yes']
)

oldpeak = st.number_input('oldpeak : ST depression induced by exercise relative to rest (ex: 2.3)')

slope = st.radio(
    "slope: the slope of the peak exercise ST segment",
    [0, 1, 2],
    captions = ['Upsloping', 'Flat', 'Downsloping']
)

ca = st.radio(
    "ca: number of major vessels (0-4) colored by flourosop",
    [0, 1, 2, 3, 4]
)

thal = st.radio(
    "thal",
    [0, 1, 2, 3]
)

st.button('Predict', on_click=startPredict)
