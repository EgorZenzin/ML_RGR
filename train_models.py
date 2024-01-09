import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.cluster import KMeans
import pickle

# Загрузка и подготовка данных
smoke_detector_data = pd.read_csv('smoke_detector.csv')
smoke_detector_data.columns = smoke_detector_data.columns.str.replace('[', '').str.replace(']', '').str.replace('<', '')
X = smoke_detector_data.drop('Fire Alarm_Yes', axis=1)
y = smoke_detector_data['Fire Alarm_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kmeans = KMeans(n_clusters=15) 
X_train_kmeans = kmeans.fit_transform(X_train)
X_test_kmeans = kmeans.transform(X_test)

# Обучение моделей
model_ml1 = LogisticRegression().fit(X_train, y_train)
model_ml4 = BaggingClassifier(base_estimator=LogisticRegression(), n_estimators=10).fit(X_train, y_train)
estimators = [('lr', LogisticRegression()), ('xgb', XGBClassifier())]
model_ml5 = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression()).fit(X_train, y_train)
model_ml3 = XGBClassifier().fit(X_train, y_train)
model_ml6 = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_ml6.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_ml6.fit(X_train, y_train, epochs=10, batch_size=32)

# Сохранение моделей
model_save_path = 'C:/Program Files/Microsoft VS Code/RGR/streamlit-models/'
pickle.dump(model_ml1, open(model_save_path + 'model_ml1.pkl', 'wb'))
pickle.dump(model_ml4, open(model_save_path + 'model_ml4.pkl', 'wb'))
pickle.dump(model_ml5, open(model_save_path + 'model_ml5.pkl', 'wb'))
model_ml3.save_model(model_save_path + 'model_ml3.json')
model_ml6.save(model_save_path + 'model_ml6.h5')
pickle.dump(kmeans, open(model_save_path + 'kmeans_model.pkl', 'wb'))