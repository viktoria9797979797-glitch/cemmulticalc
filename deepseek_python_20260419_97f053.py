# cement_predictor_universal.py - РАБОТАЕТ С ЛЮБЫМИ ЛИСТАМИ

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import os

st.set_page_config(page_title="Калькулятор прочности цемента", page_icon="🏗️", layout="wide")
st.title("🏗️ Универсальный калькулятор прочности цемента")
st.markdown("### Работает с любыми данными в правильном формате")

DATA_FILE = "List_Microsoft_Excel.xlsx"

@st.cache_data
def get_all_sheets():
    """Получает список всех листов в Excel-файле"""
    try:
        xl = pd.ExcelFile(DATA_FILE)
        return xl.sheet_names
    except:
        return []

@st.cache_data
def load_sheet_data(sheet_name):
    """Загружает данные из выбранного листа"""
    try:
        df = pd.read_excel(DATA_FILE, sheet_name=sheet_name)
        
        # Пропускаем первые строки с заголовками
        start_row = 0
        for i, row in df.iterrows():
            first_cell = str(row.iloc[0]) if pd.notna(row.iloc[0]) else ""
            if 'SiO2' in first_cell or 'SiO₂' in first_cell or first_cell == 'SiO2':
                start_row = i
                break
        
        df = pd.read_excel(DATA_FILE, sheet_name=sheet_name, skiprows=start_row)
        
        # Нормализуем колонки
        column_mapping = {
            'SiO2': 'SiO2', 'Si02': 'SiO2', 'SiO₂': 'SiO2',
            'Al2O3': 'Al2O3', 'Al203': 'Al2O3', 'Al₂O₃': 'Al2O3',
            'Fe2O3': 'Fe2O3', 'Fe203': 'Fe2O3', 'Fe₂O₃': 'Fe2O3',
            'CaO': 'CaO',
            'MgO': 'MgO',
            'SO3': 'SO3', 'SO₃': 'SO3',
            'Na2O': 'Na2O', 'Na₂O': 'Na2O',
            'Na2Oэкв': 'Na2OEQ', 'Na2O_eq': 'Na2OEQ',
            'ППП': 'ППП', 'п.п.п': 'ППП',
            'CaOсв': 'CaO_free', 'CaO_св': 'CaO_free',
            'НО': 'НО',
            '45мкм': 'sieve_45', 'остаток_45': 'sieve_45',
            'Блейн': 'Blaine', 'blaine': 'Blaine',
            'НГ': 'НГ', 'нг': 'НГ',
            'РИО': 'РИО', 'рио': 'РИО',
            'начало': 'start_set', 'Начало': 'start_set',
            'конец': 'end_set', 'Конец': 'end_set',
            'прочность 2': 'strength_2d', '2 сут': 'strength_2d',
            'прочность 28': 'strength_28d', '28 сут': 'strength_28d',
            'Известняк': 'Добавка', 'шлак': 'Добавка'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Преобразуем в числа
        for col in df.columns:
            if col not in ['Тип_цемента']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки {sheet_name}: {e}")
        return None

# Получаем список листов
sheets = get_all_sheets()

if not sheets:
    st.error(f"Файл {DATA_FILE} не найден!")
    st.stop()

# Выбор листа
selected_sheet = st.sidebar.selectbox("Выберите лист с данными", sheets)

# Загрузка данных
df = load_sheet_data(selected_sheet)

if df is None or len(df) < 10:
    st.error(f"Недостаточно данных в листе {selected_sheet} (нужно минимум 10 строк)")
    st.stop()

# Определяем доступные колонки
available_cols = [col for col in df.columns if col in [
    'SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'SO3', 'Na2O', 'Na2OEQ',
    'ППП', 'CaO_free', 'НО', 'sieve_45', 'Blaine', 'НГ', 'РИО',
    'start_set', 'end_set', 'strength_2d', 'strength_28d', 'Добавка'
]]

st.sidebar.success(f"✅ Загружено {len(df)} образцов")
st.sidebar.info(f"📊 Доступные параметры: {len(available_cols)}")

# Обучение модели
feature_cols = [col for col in available_cols if col != 'strength_28d']
if 'strength_2d' not in feature_cols:
    st.error("Нет данных о прочности на 2 сутки")
    st.stop()

X = df[feature_cols]
y = df['strength_28d']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Отображение метрик
st.subheader(f"📊 Качество модели для листа: {selected_sheet}")
col1, col2, col3 = st.columns(3)
col1.metric("R²", f"{r2:.3f}")
col2.metric("MAE", f"{mae:.2f} МПа")
col3.metric("Образцов", len(df))

st.markdown("---")

# Интерфейс ввода (динамический)
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Введите параметры")
    
    # Динамическое создание слайдеров для доступных параметров
    input_values = {}
    
    for col in feature_cols:
        if col != 'strength_2d' and df[col].notna().any():
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            default_val = float(df[col].median())
            
            if 'Blaine' in col:
                input_values[col] = st.slider(col, min_val, max_val, default_val, 50.0)
            elif 'sieve' in col:
                input_values[col] = st.slider(col, min_val, max_val, default_val, 0.2)
            else:
                input_values[col] = st.slider(col, min_val, max_val, default_val, 0.1)
    
    # Прочность 2 суток отдельно (важна)
    strength_2d_val = st.slider("Прочность 2 суток (МПа)", 
                                 float(df['strength_2d'].min()), 
                                 float(df['strength_2d'].max()), 
                                 float(df['strength_2d'].median()), 0.5)
    input_values['strength_2d'] = strength_2d_val

with col2:
    st.subheader("📈 Результаты прогноза")
    
    if st.button("🚀 Рассчитать", type="primary"):
        input_df = pd.DataFrame([input_values])[feature_cols]
        input_scaled = scaler.transform(input_df)
        predicted = model.predict(input_scaled)[0]
        
        normative = 42.5
        color = "green" if predicted >= normative else "red"
        status = "✅ Соответствует" if predicted >= normative else "❌ Не соответствует"
        
        st.markdown(f"""
        <div style='border: 3px solid {color}; border-radius: 15px; padding: 20px; text-align: center;'>
            <h1 style='color: {color};'>{predicted:.1f} МПа</h1>
            <h4>{status}</h4>
            <p>Норматив: {normative} МПа</p>
            <p style='font-size: 12px;'>Точность: ±{mae:.1f} МПа</p>
        </div>
        """, unsafe_allow_html=True)