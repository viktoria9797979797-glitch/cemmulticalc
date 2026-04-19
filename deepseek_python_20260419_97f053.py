# app.py - ПОЛНАЯ РАБОЧАЯ ВЕРСИЯ

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Калькулятор прочности цемента", page_icon="🏗️", layout="wide")

st.title("🏗️ Калькулятор прогноза прочности цемента")

DATA_FILE = "List_Microsoft_Excel.xlsx"

# Выбор типа данных
data_type = st.sidebar.selectbox(
    "Выберите тип данных",
    [
        "ЦЕМ II/А-И 42,5Б (известняк)",
        "ЦЕМ I 42,5Н (без добавок)",
        "ЦЕМ I 42,5Б (быстротвердеющий)"
    ]
)

@st.cache_data
def load_data_type1():
    """ЦЕМ II/А-И 42,5Б"""
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
        return df, "Известняк", 6.0, 21.0, "ЦЕМ II/А-И 42,5Б"
    except Exception as e:
        st.error(f"Ошибка загрузки ЦЕМ II: {e}")
        return None, None, None, None, None

@st.cache_data
def load_data_type2():
    """ЦЕМ I 42,5Н"""
    try:
        df = pd.read_excel(DATA_FILE, sheet_name="ЦЕМ I 42,5Н", skiprows=2)
        
        df.columns = [
            "SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "SO3", "ППП", "НО", 
            "CaO_free", "Na2O", "Na2OEQ", "Blaine", "sieve_45", 
            "sieve_80", "sieve_90", "РИО", "density", 
            "start_set", "end_set", "strength_2d", "strength_3d", "strength_28d", "radioactivity"
        ]
        
        df = df.drop(columns=["sieve_80", "sieve_90", "density", "radioactivity", "strength_3d"])
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna()
        df["Добавка"] = 0
        
        return df, "Добавка", 0.0, 5.0, "ЦЕМ I 42,5Н"
    except Exception as e:
        st.error(f"Ошибка загрузки ЦЕМ I 42,5Н: {e}")
        return None, None, None, None, None

@st.cache_data
def load_data_type3():
    """ЦЕМ I 42,5Б"""
    try:
        df_raw = pd.read_excel(DATA_FILE, sheet_name="ЦЕМ I 42,5Б", header=None)
        
        data_start = 0
        for i in range(len(df_raw)):
            row = df_raw.iloc[i].astype(str)
            if '95' in row.values[0] and '5' in row.values[1]:
                data_start = i
                break
        
        df = df_raw.iloc[data_start:].reset_index(drop=True)
        
        df.columns = [
            "Клинкер", "Известняк", "strength_flex_2d", "strength_flex_28d",
            "strength_2d", "strength_28d", "НГ", "start_set", "end_set",
            "РИО", "sieve_45", "Blaine", "ППП", "НО", "SiO2", "Al2O3",
            "Fe2O3", "CaO", "MgO", "SO3", "K2O", "Na2O", "Cl", "CaO_free", "Na2OEQ"
        ]
        
        df = df.drop(columns=["strength_flex_2d", "strength_flex_28d", "K2O", "Cl"])
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        df = df.dropna()
        df["Добавка"] = 100 - df["Клинкер"]
        df = df.drop(columns=["Клинкер"])
        
        return df, "Известняк", 0.0, 10.0, "ЦЕМ I 42,5Б"
    except Exception as e:
        st.error(f"Ошибка загрузки ЦЕМ I 42,5Б: {e}")
        return None, None, None, None, None

# Загрузка данных
if data_type == "ЦЕМ II/А-И 42,5Б (известняк)":
    df, additive_name, add_min, add_max, cement_name = load_data_type1()
elif data_type == "ЦЕМ I 42,5Н (без добавок)":
    df, additive_name, add_min, add_max, cement_name = load_data_type2()
else:
    df, additive_name, add_min, add_max, cement_name = load_data_type3()

if df is None:
    st.stop()

st.subheader(f"📊 Данные: {cement_name}")
st.write(f"**Найдено образцов:** {len(df)}")

# Подготовка модели
feature_cols = [col for col in df.columns if col not in ['strength_28d']]

X = df[feature_cols].copy()
y = df["strength_28d"].copy()

valid = y.notna()
for col in X.columns:
    valid = valid & X[col].notna()

X = X[valid]
y = y[valid]

if len(X) < 5:
    st.error(f"Недостаточно данных: {len(X)} строк")
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
col4.metric("Норматив", "42.5 МПа")

st.markdown("---")

# Интерфейс ввода
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Введите параметры цемента")
    
    input_values = {}
    
    # Химический состав - ВСЕ ПАРАМЕТРЫ
    st.markdown("### 🧪 Химический состав (%)")
    
    # Создаём 3 колонки для химических параметров
    chem_col1, chem_col2, chem_col3 = st.columns(3)
    
    with chem_col1:
        if 'ППП' in feature_cols:
            input_values['ППП'] = st.slider("ППП", 0.0, 10.0, float(df['ППП'].median()), 0.1)
        if 'SiO2' in feature_cols:
            input_values['SiO2'] = st.slider("SiO₂", 10.0, 30.0, float(df['SiO2'].median()), 0.1)
        if 'Al2O3' in feature_cols:
            input_values['Al2O3'] = st.slider("Al₂O₃", 0.0, 10.0, float(df['Al2O3'].median()), 0.1)
    
    with chem_col2:
        if 'Fe2O3' in feature_cols:
            input_values['Fe2O3'] = st.slider("Fe₂O₃", 0.0, 5.0, float(df['Fe2O3'].median()), 0.1)
        if 'CaO' in feature_cols:
            input_values['CaO'] = st.slider("CaO", 50.0, 70.0, float(df['CaO'].median()), 0.1)
        if 'MgO' in feature_cols:
            input_values['MgO'] = st.slider("MgO", 0.0, 3.5, float(df['MgO'].median()), 0.1)
    
    with chem_col3:
        if 'SO3' in feature_cols:
            input_values['SO3'] = st.slider("SO₃", 0.0, 4.0, float(df['SO3'].median()), 0.1)
        if 'Na2O' in feature_cols:
            input_values['Na2O'] = st.slider("Na₂O", 0.0, 1.0, float(df['Na2O'].median()), 0.01)
        if 'Na2OEQ' in feature_cols:
            input_values['Na2OEQ'] = st.slider("Na₂O экв.", 0.0, 1.5, float(df['Na2OEQ'].median()), 0.05)
    
    # Дополнительные химические параметры
    st.markdown("### 🔬 Дополнительные параметры")
    col_a, col_b = st.columns(2)
    with col_a:
        if 'НО' in feature_cols:
            input_values['НО'] = st.slider("НО", 0.0, 1.5, float(df['НО'].median()), 0.05)
        if 'CaO_free' in feature_cols:
            input_values['CaO_free'] = st.slider("CaO св.", 0.0, 1.5, float(df['CaO_free'].median()), 0.05)
    with col_b:
        if 'sieve_45' in feature_cols:
            input_values['sieve_45'] = st.slider("Остаток 45 мкм", 0.0, 10.0, float(df['sieve_45'].median()), 0.2)
        if 'Blaine' in feature_cols:
            input_values['Blaine'] = st.slider("Тонкость Блейн", 3000.0, 5000.0, float(df['Blaine'].median()), 50.0)
    
    # Добавка
    if 'Известняк' in feature_cols:
        st.markdown("### 🧱 Добавки")
        input_values['Известняк'] = st.slider("Известняк (%)", 6.0, 21.0, float(df['Известняк'].median()), 0.5)
    elif 'Добавка' in feature_cols:
        st.markdown("### 🧱 Добавки")
        input_values['Добавка'] = st.slider("Добавка (%)", 0.0, 10.0, float(df['Добавка'].median()), 0.5)
    
    # Физические свойства
    st.markdown("### ⚙️ Физические свойства")
    col_c, col_d = st.columns(2)
    with col_c:
        if 'НГ' in feature_cols:
            input_values['НГ'] = st.slider("НГ", 20.0, 40.0, float(df['НГ'].median()), 0.2)
        if 'start_set' in feature_cols:
            input_values['start_set'] = st.slider("Начало схв.", 50.0, 250.0, float(df['start_set'].median()), 5.0)
    with col_d:
        if 'РИО' in feature_cols:
            input_values['РИО'] = st.slider("РИО", 0.0, 3.0, float(df['РИО'].median()), 0.5)
        if 'end_set' in feature_cols:
            input_values['end_set'] = st.slider("Конец схв.", 100.0, 600.0, float(df['end_set'].median()), 5.0)
    
    # Прочность 2 суток
    if 'strength_2d' in feature_cols:
        st.markdown("### 💪 Прочность на ранних сроках")
        input_values['strength_2d'] = st.slider("Прочность 2 сут", 10.0, 35.0, float(df['strength_2d'].median()), 0.5)

with col2:
    st.subheader("📈 Результаты прогноза")
    
    if st.button("🚀 Рассчитать прогноз", type="primary", use_container_width=True):
        # Проверяем все ли параметры введены
        missing = [k for k in feature_cols if k not in input_values]
        if missing:
            st.warning(f"⚠️ Не все параметры введены: {', '.join(missing)}")
        else:
            input_df = pd.DataFrame([input_values])[feature_cols]
            input_scaled = scaler.transform(input_df)
            predicted = model.predict(input_scaled)[0]
            
            normative = 42.5
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
