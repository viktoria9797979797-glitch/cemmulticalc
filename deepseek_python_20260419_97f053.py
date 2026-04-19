# app.py - УНИВЕРСАЛЬНЫЙ КАЛЬКУЛЯТОР С ВЫБОРОМ ТИПА ДАННЫХ

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Калькулятор прочности цемента", page_icon="🏗️", layout="wide")

st.title("🏗️ Калькулятор прогноза прочности цемента")

# Файл с данными
DATA_FILE = "List_Microsoft_Excel.xlsx"

# Выбор типа данных
data_type = st.sidebar.selectbox(
    "Выберите тип данных",
    [
        "ЦЕМ II/А-И 42,5Б (известняк, 194 образца)",
        "ЦЕМ I 42,5Н (без добавок, 105 образцов)",
        "ЦЕМ I 42,5Б (быстротвердеющий, 100 образцов)"
    ]
)

@st.cache_data
def load_data_type1():
    """ЦЕМ II/А-И 42,5Б - известняковый цемент"""
    try:
        df = pd.read_excel(DATA_FILE, sheet_name="ЦЕМ II А-И 42,5Б", skiprows=1)
        df = df[df.iloc[:, 0] != "%"].reset_index(drop=True)
        
        df.columns = [
            "ППП", "SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "SO3", "K2O", "Na2O",
            "Na2OEQ", "НО", "CaO_free", "Шлак", "Известняк", "unused",
            "sieve_45", "Blaine", "НГ", "РИО", "start_set", "end_set", 
            "strength_2d", "strength_28d"
        ]
        
        df = df.drop(columns=["unused", "K2O", "Шлак"])
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna()
        return df, "Известняк (%)", 6.0, 21.0
    except Exception as e:
        st.error(f"Ошибка загрузки ЦЕМ II: {e}")
        return None, None, None, None

@st.cache_data
def load_data_type2():
    """ЦЕМ I 42,5Н - портландцемент без добавок"""
    try:
        # Читаем с правильной строки (данные начинаются со строки 2)
        df = pd.read_excel(DATA_FILE, sheet_name="ЦЕМ I 42,5Н", skiprows=2)
        
        # Назначаем колонки
        df.columns = [
            "SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "SO3", "ППП", "НО", 
            "CaO_free", "Na2O", "Na2OEQ", "Blaine", "sieve_45", 
            "sieve_80", "sieve_90", "РИО", "density", 
            "start_set", "end_set", "strength_2d", "strength_3d", "strength_28d", "radioactivity"
        ]
        
        # Убираем ненужные колонки
        df = df.drop(columns=["sieve_80", "sieve_90", "density", "radioactivity", "strength_3d"])
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna()
        
        # Добавляем колонку с добавкой (0 для ЦЕМ I)
        df["Добавка"] = 0
        
        return df, "Добавки (%)", 0.0, 5.0
    except Exception as e:
        st.error(f"Ошибка загрузки ЦЕМ I 42,5Н: {e}")
        return None, None, None, None

@st.cache_data
def load_data_type3():
    """ЦЕМ I 42,5Б - быстротвердеющий портландцемент"""
    try:
        # Для этого листа нужна особая обработка
        df_raw = pd.read_excel(DATA_FILE, sheet_name="ЦЕМ I 42,5Б", header=None)
        
        # Находим строку с данными (после заголовков)
        data_start = 0
        for i in range(len(df_raw)):
            row = df_raw.iloc[i].astype(str)
            if '95' in row.values[0] and '5' in row.values[1]:
                data_start = i
                break
        
        # Читаем данные
        df = df_raw.iloc[data_start:].reset_index(drop=True)
        
        # Назначаем колонки
        df.columns = [
            "Клинкер", "Известняк", "strength_flex_2d", "strength_flex_28d",
            "strength_2d", "strength_28d", "НГ", "start_set", "end_set",
            "РИО", "sieve_45", "Blaine", "ППП", "НО", "SiO2", "Al2O3",
            "Fe2O3", "CaO", "MgO", "SO3", "K2O", "Na2O", "Cl", "CaO_free", "Na2OEQ"
        ]
        
        # Убираем лишние колонки
        df = df.drop(columns=["strength_flex_2d", "strength_flex_28d", "K2O", "Cl"])
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna()
        
        # Добавка = 100 - клинкер
        df["Добавка"] = 100 - df["Клинкер"]
        df = df.drop(columns=["Клинкер"])
        
        return df, "Известняк (%)", 0.0, 10.0
    except Exception as e:
        st.error(f"Ошибка загрузки ЦЕМ I 42,5Б: {e}")
        return None, None, None, None

# Загрузка данных в зависимости от выбора
if data_type == "ЦЕМ II/А-И 42,5Б (известняк, 194 образца)":
    df, additive_name, add_min, add_max = load_data_type1()
    normative = 42.5
    cement_name = "ЦЕМ II/А-И 42,5Б"
elif data_type == "ЦЕМ I 42,5Н (без добавок, 105 образцов)":
    df, additive_name, add_min, add_max = load_data_type2()
    normative = 42.5
    cement_name = "ЦЕМ I 42,5Н"
else:
    df, additive_name, add_min, add_max = load_data_type3()
    normative = 42.5
    cement_name = "ЦЕМ I 42,5Б"

if df is None:
    st.error("Не удалось загрузить данные. Проверьте файл Excel.")
    st.stop()

st.subheader(f"📊 Данные: {cement_name}")
st.write(f"**Найдено образцов:** {len(df)}")

# Определяем доступные колонки
available_cols = [col for col in df.columns if col not in ['strength_28d']]

st.write("**Доступные параметры:**", ", ".join(available_cols[:10]) + ("..." if len(available_cols) > 10 else ""))

# Подготовка модели
feature_cols = [col for col in available_cols if col in [
    "ППП", "SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "SO3", "Na2O", 
    "Na2OEQ", "НО", "CaO_free", "Известняк", "Добавка",
    "sieve_45", "Blaine", "НГ", "РИО", "start_set", "end_set", "strength_2d"
]]

X = df[feature_cols].copy()
y = df["strength_28d"].copy()

# Удаляем строки с пропусками
valid = y.notna()
for col in X.columns:
    valid = valid & X[col].notna()

X = X[valid]
y = y[valid]

if len(X) < 5:
    st.error(f"Недостаточно данных после очистки: {len(X)} строк")
    st.stop()

# Обучение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Метрики
col1, col2, col3, col4 = st.columns(4)
col1.metric("R²", f"{r2:.3f}")
col2.metric("MAE", f"{mae:.2f} МПа")
col3.metric("Образцов", len(X))
col4.metric("Норматив", f"{normative} МПа")

st.markdown("---")

# Интерфейс ввода
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Введите параметры цемента")
    
    input_values = {}
    
    # Химический состав
    st.markdown("### 🧪 Химический состав (%)")
    chem_cols = ['SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'SO3', 'Na2OEQ', 'ППП', 'CaO_free', 'НО']
    available_chem = [c for c in chem_cols if c in feature_cols]
    
    if available_chem:
        cols = st.columns(min(3, len(available_chem)))
        for i, col_name in enumerate(available_chem):
            with cols[i % 3]:
                min_val = float(df[col_name].min())
                max_val = float(df[col_name].max())
                default_val = float(df[col_name].median())
                input_values[col_name] = st.slider(col_name, min_val, max_val, default_val, 0.1, format="%.2f")
    
    # Физические характеристики
    st.markdown("### ⚙️ Физические характеристики")
    phys_cols = ['Blaine', 'sieve_45', 'НГ', 'РИО']
    available_phys = [c for c in phys_cols if c in feature_cols]
    
    if available_phys:
        cols = st.columns(min(2, len(available_phys)))
        for i, col_name in enumerate(available_phys):
            with cols[i % 2]:
                min_val = float(df[col_name].min())
                max_val = float(df[col_name].max())
                default_val = float(df[col_name].median())
                if 'Blaine' in col_name:
                    input_values[col_name] = st.slider(col_name, min_val, max_val, default_val, 50.0, format="%.0f")
                else:
                    input_values[col_name] = st.slider(col_name, min_val, max_val, default_val, 0.2, format="%.1f")
    
    # Сроки схватывания
    st.markdown("### ⏱️ Сроки схватывания")
    time_cols = ['start_set', 'end_set']
    available_time = [c for c in time_cols if c in feature_cols]
    
    if available_time:
        cols = st.columns(2)
        for i, col_name in enumerate(available_time):
            with cols[i % 2]:
                min_val = float(df[col_name].min())
                max_val = float(df[col_name].max())
                default_val = float(df[col_name].median())
                input_values[col_name] = st.slider(col_name, min_val, max_val, default_val, 5.0, format="%.0f")
    
    # Прочность 2 суток
    if 'strength_2d' in feature_cols:
        st.markdown("### 💪 Прочность на ранних сроках")
        min_val = float(df['strength_2d'].min())
        max_val = float(df['strength_2d'].max())
        default_val = float(df['strength_2d'].median())
        input_values['strength_2d'] = st.slider("Прочность 2 суток (МПа)", min_val, max_val, default_val, 0.5, format="%.1f")
    
    # Добавка
    additive_col = None
    if 'Известняк' in feature_cols:
        additive_col = 'Известняк'
        add_label = "Известняк (%)"
    elif 'Добавка' in feature_cols:
        additive_col = 'Добавка'
        add_label = additive_name
    
    if additive_col:
        st.markdown("### 🧱 Добавки")
        min_val = add_min if add_min else float(df[additive_col].min())
        max_val = add_max if add_max else float(df[additive_col].max())
        default_val = float(df[additive_col].median())
        input_values[additive_col] = st.slider(add_label, min_val, max_val, default_val, 0.5, format="%.1f")

with col2:
    st.subheader("📈 Результаты прогноза")
    
    if st.button("🚀 Рассчитать прогноз", type="primary", use_container_width=True):
        missing = [k for k in feature_cols if k not in input_values]
        if missing:
            st.warning(f"⚠️ Не все параметры введены: {', '.join(missing)}")
        else:
            input_df = pd.DataFrame([input_values])[feature_cols]
            input_scaled = scaler.transform(input_df)
            predicted = model.predict(input_scaled)[0]
            
            color = "green" if predicted >= normative else "red"
            status = "✅ Соответствует" if predicted >= normative else "❌ Не соответствует"
            
            st.markdown(f"""
            <div style='border: 3px solid {color}; border-radius: 15px; padding: 20px; text-align: center; background-color: #f8f9fa;'>
                <h1 style='color: {color}; font-size: 48px;'>{predicted:.1f} МПа</h1>
                <h4>{status} {cement_name}</h4>
                <p>Норматив: <b>{normative} МПа</b></p>
                <p style='font-size: 12px;'>Точность: ±{mae:.1f} МПа</p>
            </div>
            """, unsafe_allow_html=True)
            
            progress = min(max((predicted - 35) / 20, 0), 1)
            st.progress(progress)

# Боковая панель
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 О модели")
    st.markdown(f"""
    - **Тип:** {cement_name}
    - **Образцов:** {len(X)}
    - **Признаков:** {len(feature_cols)}
    - **R²:** {r2:.3f}
    - **MAE:** {mae:.2f} МПа
    """)
