import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from xgboost import XGBClassifier
from sklearn.ensemble import BaggingClassifier, StackingClassifier
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, Dropout
from pandas.plotting import scatter_matrix
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

model_save_path = 'streamlit-models/'

smoke_detector_data = pd.read_csv('smoke_detector.csv')
smoke_detector_data.columns = smoke_detector_data.columns.str.replace('[', '').str.replace(']', '').str.replace('<', '')
X = smoke_detector_data.drop('Fire Alarm_Yes', axis=1)
y = smoke_detector_data['Fire Alarm_Yes']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def load_models():
    model_ml1 = pickle.load(open(model_save_path + 'model_ml1.pkl', 'rb'))
    model_ml4 = pickle.load(open(model_save_path + 'model_ml4.pkl', 'rb'))
    model_ml5 = pickle.load(open(model_save_path + 'model_ml5.pkl', 'rb'))
    model_ml3 = XGBClassifier()
    model_ml3.load_model(model_save_path + 'model_ml3.json')
    model_ml6 = load_model(model_save_path + 'model_ml6.h5')
    model_ml2 = pickle.load(open(model_save_path + 'kmeans_model.pkl', 'rb'))
    return model_ml1, model_ml3, model_ml4, model_ml5, model_ml6, model_ml2

st.markdown("""
<style>
.sidebar .sidebar-content {
    background-color: #f1f3f6;
}
h1 {
    color: #0e1117;
}
</style>
""", unsafe_allow_html=True)

# Сайдбар для навигации
page = st.sidebar.radio(
    "Выберите страницу:",
    ("Информация о разработчике", "Информация о наборе данных", "Визуализации данных", "Предсказание модели ML")
)

smoke_detector_data = pd.read_csv('smoke_detector.csv')

# Функции для каждой страницы
def page_developer_info():
    st.title("Информация о разработчике")
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Контактная информация")
        st.write("ФИО: Зензин Егор Николаевич")
        st.write("Номер учебной группы: ФИТ-222")
    
    with col2:
        st.header("Фотография")
        st.image("Egor.jpg", width=250)
    
    st.header("Тема РГР")
    st.write("Разработка Web-приложения для инференса моделей ML и анализа данных")

def page_dataset_info():
    st.title("Информация о наборе данных")

    st.markdown("""
    ## Описание Датасета smoke_detector
    **Файл датасета:** `smoke_detector.csv`

    **Описание:**
    Данный датасет содержит статистическую информацию о качестве воздуха и параметрах окружающей среды. Информация по каждому столбцу:

    - `Unnamed 0`: Номер строки            
    - `UTC`: Временная метка в формате, представляющая момент времени в секундах с начала эпохи.
    - `Temperature[C]`: Измеренная температура в градусах Цельсия.
    - `Humidity[%]`: Уровень влажности в процентах.
    - `TVOC[ppb]`: Количество летучих органических соединений в частях на миллиард.
    - `eCO2[ppm]`: Количество углекислого газа в частях на миллион.
    - `Raw H2`: Измеренное количество водорода в сыром формате.
    - `Raw Ethanol`: Измеренное количество этанола в сыром формате.
    - `Pressure[hPa]`: Атмосферное давление в гектопаскалях.
    - `PM1.0`: Концентрация частиц с диаметром 1.0 микрометра и меньше.
    - `PM2.5`: Концентрация частиц с диаметром 2.5 микрометра и меньше.
    - `NC0.5`: Количество частиц с диаметром 0.5 микрометра и больше.
    - `NC1.0`: Количество частиц с диаметром 1.0 микрометра и больше.
    - `NC2.5`: Количество частиц с диаметром 2.5 микрометра и больше.
    - `CNT`: Общее количество частиц
    - `Fire Alarm_Yes`: Индикатор сработавшей сигнализации о пожаре (1 - да, 0 - нет). 
                
    **Особенности предобработки данных:**
    - Удаление лишних столбцов, таких как, 'Unnamed 0'.
    - Замена пропущенных значений.
    - Очистка данных, с помощью удаления дупликатов.
    - Преобразование категориальных признаков.
    """)

def page_data_visualization():
    st.title("Визуализации данных smoke_detector")

    # Выбираем численные признаки для визуализации
    numeric_features = smoke_detector_data.select_dtypes(include=["float64", "int64"]).columns

    # Гистограмма влажности
    plt.figure(figsize=(10, 6))
    sns.histplot(smoke_detector_data['Humidity[%]'], bins=10, kde=True)
    plt.title('Распределение влажности')
    plt.xlabel('Влажность (%)')
    plt.ylabel('Частота')
    plt.grid(True)
    st.pyplot(plt)

    # Точечный график давления в зависимости от концентрации CO2
    plt.figure(figsize=(10, 6))
    plt.scatter(smoke_detector_data['eCO2[ppm]'], smoke_detector_data['Pressure[hPa]'])
    plt.title('Давление в зависимости от концентрации CO2')
    plt.xlabel('eCO2 (ppm)')
    plt.ylabel('Давление (hPa)')
    plt.grid(True)
    st.pyplot(plt)

    # Выбросы
    plt.figure(figsize=(10, 6))
    plt.title('Выбросы Humidity[%]')
    sns.boxplot(y='Humidity[%]',data=smoke_detector_data)
    st.pyplot(plt)

    # Тепловая карта корреляции
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(12, 14))
    correlation_matrix = smoke_detector_data[numeric_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title('Тепловая карта корелляции', fontsize=20)
    plt.savefig("heatmap.png")
    st.image("heatmap.png")
        

# Функция для загрузки моделей
def load_models():
    model_ml1 = pickle.load(open(model_save_path + 'model_ml1.pkl', 'rb'))
    model_ml4 = pickle.load(open(model_save_path + 'model_ml4.pkl', 'rb'))
    model_ml5 = pickle.load(open(model_save_path + 'model_ml5.pkl', 'rb'))
    model_ml3 = XGBClassifier()
    model_ml3.load_model(model_save_path + 'model_ml3.json')
    model_ml6 = load_model(model_save_path + 'model_ml6.h5')
    model_ml2 = pickle.load(open(model_save_path + 'kmeans_model.pkl', 'rb'))
    return model_ml1, model_ml3, model_ml4, model_ml5, model_ml6, model_ml2

def page_ml_prediction():
    st.title("Предсказания моделей машинного обучения")

    # Виджет для загрузки файла
    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")

    # Интерактивный ввод данных, если файл не загружен
    if uploaded_file is None:
        st.subheader("Введите данные для предсказания:")

        # Интерактивные поля для ввода данных
        input_data = {}
        all_columns = smoke_detector_data.columns.tolist()
        feature_names = all_columns
        feature_names.remove("Fire Alarm_Yes")
        for feature in feature_names:
            input_data[feature] = st.number_input(f"{feature}", min_value=0.0, max_value=100000.0, value=50.0)

        if st.button('Сделать предсказание'):
            # Загрузка моделей
            model_ml1, model_ml3, model_ml4, model_ml5, model_ml6, model_ml2 = load_models()

            input_df = pd.DataFrame([input_data])
            
            st.write("Входные данные:", input_df)

            # Используем масштабировщик, обученный на обучающих данных
            scaler = StandardScaler().fit(X_train)
            scaled_input = scaler.transform(input_df)

            # Делаем предсказания
            prediction_ml1 = model_ml1.predict(scaled_input)
            prediction_ml3 = model_ml3.predict(scaled_input)
            prediction_ml4 = model_ml4.predict(scaled_input)
            prediction_ml5 = model_ml5.predict(scaled_input)
            prediction_ml6 = (model_ml6.predict(scaled_input) > 0.5).astype(int)

            # Вывод результатов
            st.success(f"Результат предсказания LogisticRegression: {prediction_ml1[0]}")
            st.success(f"Результат предсказания XGBClassifier: {prediction_ml3[0]}")
            st.success(f"Результат предсказания BaggingClassifier: {prediction_ml4[0]}")
            st.success(f"Результат предсказания StackingClassifier: {prediction_ml5[0]}")
            st.success(f"Результат предсказания нейронной сети Tensorflow: {prediction_ml6[0]}")
    else:
        try:
            model_ml2 = pickle.load(open(model_save_path + 'kmeans_model.pkl', 'rb'))
            model_ml1 = pickle.load(open(model_save_path + 'model_ml1.pkl', 'rb'))
            model_ml4 = pickle.load(open(model_save_path + 'model_ml4.pkl', 'rb'))
            model_ml5 = pickle.load(open(model_save_path + 'model_ml5.pkl', 'rb'))
            model_ml3 = XGBClassifier()
            model_ml3.load_model(model_save_path + 'model_ml3.json')
            model_ml6 = load_model(model_save_path + 'model_ml6.h5')

            # Сделать предсказания на тестовых данных
            cluster_labels = model_ml2.predict(X_test)
            predictions_ml1 = model_ml1.predict(X_test)
            predictions_ml4 = model_ml4.predict(X_test)
            predictions_ml5 = model_ml5.predict(X_test)
            predictions_ml3 = model_ml3.predict(X_test)
            predictions_ml6 = model_ml6.predict(X_test).round() # Округление для нейронной сети

            # Оценить результаты
            rand_score_ml2 = rand_score(y_test, cluster_labels)
            accuracy_ml1 = accuracy_score(y_test, predictions_ml1)
            accuracy_ml4 = accuracy_score(y_test, predictions_ml4)
            accuracy_ml5 = accuracy_score(y_test, predictions_ml5)
            accuracy_ml3 = accuracy_score(y_test, predictions_ml3)
            accuracy_ml6 = accuracy_score(y_test, predictions_ml6)

            st.success(f"rand_score KMeans: {rand_score_ml2}")
            st.success(f"Точность LogisticRegression: {accuracy_ml1}")
            st.success(f"Точность XGBClassifier: {accuracy_ml4}")
            st.success(f"Точность BaggingClassifier: {accuracy_ml5}")
            st.success(f"Точность StackingClassifier: {accuracy_ml3}")
            st.success(f"Точность нейронной сети Tensorflow: {accuracy_ml6}")
        except Exception as e:
            st.error(f"Произошла ошибка при чтении файла: {e}")


if page == "Информация о разработчике":
    page_developer_info()
elif page == "Информация о наборе данных":
    page_dataset_info()
elif page == "Визуализации данных":
    page_data_visualization()
elif page == "Предсказание модели ML":
    page_ml_prediction()
