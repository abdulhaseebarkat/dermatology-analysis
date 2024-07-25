import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import CategoricalCrossentropy


url = "D:\\Machine Learning\\dematology-dieses\\dataset\\dermatology.data"
data = pd.read_csv(url, header=None)


data.replace('?', np.nan, inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)  

imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)


X = data_imputed[:, :-1]
y = data_imputed[:, -1]


scaler = StandardScaler()
X = scaler.fit_transform(X)


smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)


encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


def build_and_train_model(optimizer, loss):
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=2)
    return model, history


optimizers = [Adam(), SGD()]
losses = [CategoricalCrossentropy()]


results = []
for optimizer in optimizers:
    for loss in losses:
        model, history = build_and_train_model(optimizer, loss)
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        recall = recall_score(y_test_classes, y_pred_classes, average='macro')
        precision = precision_score(y_test_classes, y_pred_classes, average='macro')
        f1 = f1_score(y_test_classes, y_pred_classes, average='macro')
        conf_matrix = confusion_matrix(y_test_classes, y_pred_classes)
        
        results.append({
            'optimizer': optimizer,
            'loss': loss,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'history': history
        })


best_result = max(results, key=lambda x: x['accuracy'])
print(f"Best Model - Optimizer: {best_result['optimizer']}, Loss: {best_result['loss']}")
print(f"Accuracy: {best_result['accuracy']}")
print(f"Recall: {best_result['recall']}")
print(f"Precision: {best_result['precision']}")
print(f"F1 Score: {best_result['f1']}")
print("Confusion Matrix:")
print(best_result['confusion_matrix'])


plt.figure(figsize=(10, 8))
sns.heatmap(best_result['confusion_matrix'], annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(best_result['history'].history['accuracy'], label='Train Accuracy')
plt.plot(best_result['history'].history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(best_result['history'].history['loss'], label='Train Loss')
plt.plot(best_result['history'].history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.show()
