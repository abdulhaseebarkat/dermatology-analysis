import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

url = "./dermatology.data"
data = pd.read_csv(url, header=None)

data.replace('?', np.nan, inplace=True)

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data.fillna(data.median(), inplace=True)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(), metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=12, batch_size=10, validation_split=0.2, verbose=2)

model.save('dermatology_model.h5')


y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
accuracy = accuracy_score(y_test_classes, y_pred_classes)*100
print(f"Accuracy: {accuracy:.2f}%")

model = load_model('dermatology_model.h5')


def get_user_input():
    attributes = [
        {"name": "erythema", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "scaling", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "definite borders", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "itching", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "koebner phenomenon", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "polygonal papules", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "follicular papules", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "oral mucosal involvement", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "knee and elbow involvement", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "scalp involvement", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "family history", "options": ["none", "present"]},  # This is a binary attribute.
        {"name": "melanin incontinence", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "eosinophils in the infiltrate", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "PNL infiltrate", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "fibrosis of the papillary dermis", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "exocytosis", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "acanthosis", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "hyperkeratosis", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "parakeratosis", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "clubbing of the rete ridges", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "elongation of the rete ridges", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "thinning of the suprapapillary epidermis", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "spongiform pustule", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "munro microabcess", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "focal hypergranulosis", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "disappearance of the granular layer", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "vacuolisation and damage of basal layer", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "spongiosis", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "saw-tooth appearance of retes", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "follicular horn plug", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "perifollicular parakeratosis", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "inflammatory monoluclear infiltrate", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "band-like infiltrate", "options": ["not present", "not so much present", "a little more present", "present"]},
        {"name": "age", "options": ["enter your age"]} 
    ]

    responses = []
    for attribute in attributes:
        print(attribute["name"].capitalize())
        print("----------")
        if attribute["name"] == "age":
            response = int(input("Please enter your age: "))
        else:
            for i, option in enumerate(attribute["options"]):
                print(f"{i}. {option}")
            response = int(input("Please select: "))
        responses.append(response)
        print()
    return responses

user_responses = get_user_input()

user_inputs = np.array(user_responses)  
user_inputs = user_inputs.reshape(1, -1) 
prediction = model.predict(user_inputs)
print(prediction)
predicted_class = np.argmax(prediction, axis=1)
confidence = np.max(prediction) * 100

print(f"The system predicted that you have: {predicted_class} with accuracy {confidence:.2f}%")