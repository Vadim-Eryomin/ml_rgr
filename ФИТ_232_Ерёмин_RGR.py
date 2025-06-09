import joblib
import numpy
import streamlit as st

st.sidebar.title("Навигация")

page = st.sidebar.radio('', [
    "О разработчике",
    "О датасете",
    "Визуализации",
    "Получение предсказания"
])


if page == "О разработчике":
    st.title("Информация о разработчике")
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image("./rgr/photo.jpg", caption="Моя фоточка", width=150)
    
    with col2:
        st.header("Ерёмин Вадим Сергеевич")
        st.subheader("ФИТиКС, ФИТ-232/1")
        st.write("**Контакты:** vadimeryomin22@gmail.com")


elif page == "О датасете":
    st.title("Информация о датасете")
    
    # Пример описания
    st.write("""
    ### Описание
    Датасет содержит информацию о матчах в CSGO
    
    ### Структура данных
    #### Описание 
    - **time_left** Время оставшееся до конца матча
    - **сt_health** Здоровье у команды Conter-Terrorist
    - **t_health** Здоровье у команды Terrorist
    - **ct_armor** Броня у команды Contrer-Terrorist
    - **ct_players_alive** Количество живых у команды Conter-Terrorist
    - **t_players_alive** Количество живых у команды Terrorist
    - **Целевая переменная bomb_planted:** Заложена ли бомба
    #### Предобработка
    - Пропуски установлены по средним
    - Из генеральной выборки выделены наиболее значимые признаки
    - Выбросов нет
    """)
    
    # Пример загрузки данных
    if st.button("Показать пример данных"):
        import pandas as pd
        import numpy as np
        
        data = pd.read_csv('csgo_less.csv')
        st.dataframe(data.head(10))

elif page == "Визуализации":
    st.title("Анализ и визуализации")
    st.write("""
    ### Визуализации
    """)
    
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import seaborn as sn

    data = pd.read_csv('csgo_less.csv')[:1000]
    st.scatter_chart(data, x='time_left', y='ct_health', x_label='Времени осталось, сек', y_label='Здоровье у команды Conter-Terrorist', color='bomb_planted')
    
    pair = sn.pairplot(data, y_vars=['time_left'], x_vars=['ct_health', 't_health', 'bomb_planted'])
    st.pyplot(pair)

    fig, ax = plt.subplots(figsize=(10, 8))
    heat = sn.heatmap(data.corr(), vmin=-1, vmax=1, ax=ax)
    st.pyplot(fig)

    plt.figure(figsize=(10, 5))
    plt.bar(data['time_left'], data['ct_players_alive'], width=0.4, label='CT Alive')
    plt.bar(data['time_left'] + 0.4, data['t_players_alive'], width=0.4, label='T Alive')
    plt.xlabel('Time Left')
    plt.ylabel('Players Alive')
    plt.title('Players Alive by Team Over Time')
    plt.legend()
    st.pyplot(plt)


# Страница 4: Получение предсказания
elif page == "Получение предсказания":
    # Загрузка моделей
    lin = joblib.load('models/lin.l')
    grad = joblib.load('models/grad.l')
    catboost = joblib.load('models/catboost.l')
    forest = joblib.load('models/forest.l')
    stacking = joblib.load('models/stacking.l')
    nn = joblib.load('models/nn.l')

    st.title("Прогнозирование")
    
    # Табы для выбора режима предсказания
    tab1, tab2 = st.tabs(["Одиночное предсказание", "Пакетное предсказание (файл)"])
    
    with tab1:
        st.subheader("Введите параметры вручную")
        col1, col2 = st.columns(2)
        with col1:
            feature0 = st.slider("Время до конца матча", 0.0, 180.0, 1.0, 1.0)
            feature1 = st.slider("Суммарное здоровье команды Counter-Terrorist", 0.0, 500.0, 5.0, 5.0)
            feature2 = st.slider("Суммарное здоровье команды Terrorist", 0.0, 500.0, 5.0, 5.0)
        with col2:
            feature3 = st.slider("Суммарная броня команды Terrorist", 0.0, 500.0, 5.0, 5.0)
            feature4 = st.slider("Выживших из команды Counter-Terrorist", 0.0, 5.0, 1.0, 1.0)
            feature5 = st.slider("Выживших из команды Terrorist", 0.0, 5.0, 1.0, 1.0)
        
        if st.button("Получить предсказание", key="single_pred"):
            input_data = numpy.asarray([feature0, feature1, feature2, feature3, feature4, feature5]).reshape(1, -1)
            
            st.success(f"Линейная модель: {'Бомба заложена' if lin.predict(input_data)[0] == 1 else 'Бомба не заложена'}")
            st.success(f"Градиентный бустинг: {'Бомба заложена' if grad.predict(input_data)[0] == 1 else 'Бомба не заложена'}")
            st.success(f"CatBoost: {'Бомба заложена' if catboost.predict(input_data)[0] == 1 else 'Бомба не заложена'}")
            st.success(f"Random Forest: {'Бомба заложена' if forest.predict(input_data)[0] == 1 else 'Бомба не заложена'}")
            st.success(f"Stacking модель: {'Бомба заложена' if stacking.predict(input_data)[0] == 1 else 'Бомба не заложена'}")
            st.success(f"Нейросеть: {'Бомба заложена' if nn.predict(input_data)[0] == 1 else 'Бомба не заложена'}")
    
    with tab2:
        st.subheader("Загрузите файл с данными для предсказания")
        st.write("Файл должен содержать следующие колонки:")
        st.write("- time_left (Время до конца матча)")
        st.write("- ct_health (Суммарное здоровье CT)")
        st.write("- t_health (Суммарное здоровье T)")
        st.write("- t_armor (Суммарная броня T)")
        st.write("- ct_alive (Выжившие CT)")
        st.write("- t_alive (Выжившие T)")
        
        uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])

        if uploaded_file is not None:
            try:
                import pandas as pd
                # Чтение файла
                df = pd.read_csv(uploaded_file)
                
                # Проверка наличия нужных колонок
                required_columns = ['time_left','ct_health','t_health','ct_armor','ct_players_alive','t_players_alive']
                if not all(col in df.columns for col in required_columns):
                    st.error("Файл не содержит всех необходимых колонок!")
                    for i in required_columns:
                        if  i not in df.columns:
                            st.error(i)
                else:
                    st.success("Файл успешно загружен!")
                    st.dataframe(df.head())
                    
                    if st.button("Выполнить пакетное предсказание", key="batch_pred"):
                        # Подготовка данных
                        X = df[required_columns].values
                        
                        # Получение предсказаний
                        df['lin_pred'] = lin.predict(X)
                        df['grad_pred'] = grad.predict(X)
                        df['catboost_pred'] = catboost.predict(X)
                        df['forest_pred'] = forest.predict(X)
                        df['stacking_pred'] = stacking.predict(X)
                        df['nn_pred'] = nn.predict(X)
                        
                        # Преобразование в читаемый формат
                        for model in ['lin', 'grad', 'catboost', 'forest', 'stacking', 'nn']:
                            df[f'{model}_pred'] = df[f'{model}_pred'].map({1: 'Бомба заложена', 0: 'Бомба не заложена'})
                        
                        # Отображение результатов
                        st.subheader("Результаты предсказаний")
                        st.dataframe(df)
                        
                        # Кнопка для скачивания результатов
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Скачать результаты в CSV",
                            data=csv,
                            file_name='predictions.csv',
                            mime='text/csv'
                        )
                        
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {str(e)}")