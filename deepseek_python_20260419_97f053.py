# cement_predictor_universal.py - РАБОТАЕТ СО ВСЕМИ ВАШИМИ ЛИСТАМИ

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import re

st.set_page_config(page_title="Калькулятор прочности цемента", page_icon="🏗️", layout="wide")
st.title("🏗️ Калькулятор прогноза прочности цемента")
st.markdown("### Поддерживаются все типы цемента: ЦЕМ I, ЦЕМ II")

DATA_FILE = "List_Microsoft_Excel.xlsx"

# Словарь для нормализации названий колонок
def normalize_columns(df):
    """Приводит названия колонок к стандартному виду"""
    new_columns = {}
    
    for col in df.columns:
        col_str = str(col).strip().lower()
        col_clean = re.sub(r'[^\wа-я]', '', col_str)  # убираем спецсимволы
        
        # Прочность 2 суток
        if 'прочность2' in col_clean or '2сут' in col_clean or '2дн' in col_clean:
            new_columns[col] = 'strength_2d'
        elif 'прочность' in col_clean and '2' in col_clean:
            new_columns[col] = 'strength_2d'
        elif 'прочность2сут' in col_clean:
            new_columns[col] = 'strength_2d'
        
        # Прочность 28 суток
        elif 'прочность28' in col_clean or '28сут' in col_clean or '28дн' in col_clean:
            new_columns[col] = 'strength_28d'
        elif 'прочность' in col_clean and '28' in col_clean:
            new_columns[col] = 'strength_28d'
        elif 'прочность28сут' in col_clean:
            new_columns[col] = 'strength_28d'
        
        # Химический состав
        elif col_clean in ['sio2', 'кремнезём', 'кремнезем']:
            new_columns[col] = 'SiO2'
        elif col_clean in ['al2o3', 'al2oз', 'глинозём', 'глинозем']:
            new_columns[col] = 'Al2O3'
        elif col_clean in ['fe2o3', 'fe2oз', 'оксиджелеза']:
            new_columns[col] = 'Fe2O3'
        elif col_clean in ['cao', 'оксидкальция']:
            new_columns[col] = 'CaO'
        elif col_clean in ['mgo', 'оксидмагния']:
            new_columns[col] = 'MgO'
        elif col_clean in ['so3', 'сульфаты']:
            new_columns[col] = 'SO3'
        elif col_clean in ['na2o', 'na2oэ', 'na2oэкв', 'na2oэкв.']:
            new_columns[col] = 'Na2OEQ'
        elif col_clean in ['ппп', 'потери']:
            new_columns[col] = 'ППП'
        elif 'caoсв' in col_clean or 'cao_free' in col_clean:
            new_columns[col] = 'CaO_free'
        elif col_clean in ['но']:
            new_columns[col] = 'НО'
        
        # Физические характеристики
        elif 'блейн' in col_clean or 'blaine' in col_clean:
            new_columns[col] = 'Blaine'
        elif '45мкм' in col_clean or '45мк' in col_clean:
            new_columns[col] = 'sieve_45'
        elif col_clean in ['нг']:
            new_columns[col] = 'НГ'
        elif col_clean in ['рио']:
            new_columns[col] = 'РИО'
        
        # Сроки схватывания
        elif 'начало' in col_clean:
            new_columns[col] = 'start_set'
        elif 'конец' in col_clean:
            new_columns[col] = 'end_set'
        
        # Добавки
        elif 'известняк' in col_clean:
            new_columns[col] = 'Добавка'
        elif 'шлак' in col_clean:
            new_columns[col] = 'Добавка'
        elif col_clean == 'клинкер':
            new_columns[col] = 'Клинкер'
    
    df = df.rename(columns=new_columns)
    return df

@st.cache_data
def load_sheet_data(sheet_name):
    """Загружает данные из выбранного листа"""
    try:
        # Пробуем прочитать файл
        df = pd.read_excel(DATA_FILE, sheet_name=sheet_name)
        
        # Для листа ЦЕМ I 42,5Б - особая структура (данные в строках, а не колонках)
        if 'ЦЕМ I 42,5Б' in sheet_name:
            # Транспонируем, чтобы параметры стали колонками
            df = df.T
            df.columns = df.iloc[0]
            df = df[1:]
            df = df.reset_index(drop=True)
        
        # Нормализуем названия колонок
        df = normalize_columns(df)
        
        # Преобразуем в числа
        for col in df.columns:
            if col not in ['Тип_цемента', 'Клинкер']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Удаляем строки с пропусками в целевой переменной
        if 'strength_28d' in df.columns:
            df = df.dropna(subset=['strength_28d'])
        
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки {sheet_name}: {e}")
        return None

# Получаем список листов
try:
    sheets = pd.ExcelFile(DATA_FILE).sheet_names
except:
    st.error(f"Файл {DATA_FILE} не найден!")
    st.stop()

# Выбор листа
st.sidebar.header("📁 Выбор данных")
selected_sheet = st.sidebar.selectbox("Выберите лист с данными", sheets)

# Загрузка данных
df = load_sheet_data(selected_sheet)

if df is None or len(df) < 5:
    st.error(f"Недостаточно данных в листе '{selected_sheet}'")
    st.stop()

# Показываем найденные колонки
st.subheader(f"📊 Данные из листа: {selected_sheet}")
st.write(f"**Найдено образцов:** {len(df)}")

# Определяем доступные колонки
available_cols = [col for col in df.columns if col not in ['Клинкер', 'Тип_цемента']]

st.write("**Доступные параметры:**", ", ".join(available_cols))

# Проверяем наличие целевой переменной
if 'strength_28d' not in df.columns:
    st.error("❌ Не найдена колонка с прочностью 28 суток!")
    st.stop()

if 'strength_2d' not in df.columns:
    st.warning("⚠️ Нет данных о прочности на 2 сутки. Будут использованы только химические параметры.")

# Подготовка признаков
feature_cols = [col for col in available_cols if col != 'strength_28d']

X = df[feature_cols].copy()
y = df['strength_28d'].copy()

# Удаляем строки с пропусками
valid = y.notna()
for col in X.columns:
    valid = valid & X[col].notna()

X = X[valid]
y = y[valid]

if len(X) < 5:
    st.error(f"После очистки осталось только {len(X)} строк")
    st.stop()

# Обучение модели
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Отображение метрик
col1, col2, col3, col4 = st.columns(4)
col1.metric("R²", f"{r2:.3f}")
col2.metric("MAE", f"{mae:.2f} МПа")
col3.metric("Образцов", len(X))
col4.metric("Признаков", len(feature_cols))

st.markdown("---")

# Интерфейс ввода
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Введите параметры цемента")
    
    input_values = {}
    
    # Группировка параметров
    st.markdown("### 🧪 Химический состав")
    
    chem_cols = ['SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'SO3', 'Na2OEQ', 'ППП', 'CaO_free', 'НО']
    available_chem = [c for c in chem_cols if c in feature_cols]
    
    if available_chem:
        cols = st.columns(min(3, len(available_chem)))
        for i, col_name in enumerate(available_chem):
            with cols[i % 3]:
                min_val = float(X[col_name].min())
                max_val = float(X[col_name].max())
                default_val = float(X[col_name].median())
                input_values[col_name] = st.slider(col_name, min_val, max_val, default_val, 0.1, format="%.2f")
    
    st.markdown("### ⚙️ Физические характеристики")
    
    phys_cols = ['Blaine', 'sieve_45', 'НГ', 'РИО']
    available_phys = [c for c in phys_cols if c in feature_cols]
    
    if available_phys:
        cols = st.columns(min(2, len(available_phys)))
        for i, col_name in enumerate(available_phys):
            with cols[i % 2]:
                min_val = float(X[col_name].min())
                max_val = float(X[col_name].max())
                default_val = float(X[col_name].median())
                if 'Blaine' in col_name:
                    input_values[col_name] = st.slider(col_name, min_val, max_val, default_val, 50.0, format="%.0f")
                else:
                    input_values[col_name] = st.slider(col_name, min_val, max_val, default_val, 0.2, format="%.1f")
    
    st.markdown("### ⏱️ Сроки схватывания")
    
    time_cols = ['start_set', 'end_set']
    available_time = [c for c in time_cols if c in feature_cols]
    
    if available_time:
        cols = st.columns(2)
        for i, col_name in enumerate(available_time):
            with cols[i % 2]:
                min_val = float(X[col_name].min())
                max_val = float(X[col_name].max())
                default_val = float(X[col_name].median())
                input_values[col_name] = st.slider(col_name, min_val, max_val, default_val, 5.0, format="%.0f")
    
    # Прочность 2 суток
    if 'strength_2d' in feature_cols:
        st.markdown("### 💪 Прочность на ранних сроках")
        min_val = float(X['strength_2d'].min())
        max_val = float(X['strength_2d'].max())
        default_val = float(X['strength_2d'].median())
        input_values['strength_2d'] = st.slider("Прочность 2 суток (МПа)", min_val, max_val, default_val, 0.5, format="%.1f")
    
    # Добавка
    if 'Добавка' in feature_cols:
        st.markdown("### 🧱 Добавки")
        min_val = float(X['Добавка'].min())
        max_val = float(X['Добавка'].max())
        default_val = float(X['Добавка'].median())
        input_values['Добавка'] = st.slider("Содержание добавки (%)", min_val, max_val, default_val, 0.5, format="%.1f")

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
            
            normative = 42.5
            color = "green" if predicted >= normative else "red"
            status = "✅ Соответствует" if predicted >= normative else "❌ Не соответствует"
            
            st.markdown(f"""
            <div style='border: 3px solid {color}; border-radius: 15px; padding: 20px; text-align: center; background-color: #f8f9fa;'>
                <h1 style='color: {color}; font-size: 48px;'>{predicted:.1f} МПа</h1>
                <h4>{status}</h4>
                <p>Норматив: <b>{normative} МПа</b></p>
                <p style='font-size: 12px; color: gray;'>Точность прогноза: ±{mae:.1f} МПа</p>
            </div>
            """, unsafe_allow_html=True)
            
            progress = min(max((predicted - 35) / 20, 0), 1)
            st.progress(progress)
            
            if predicted < normative:
                st.warning("💡 **Рекомендации:**")
                if 'strength_2d' in input_values and input_values['strength_2d'] < 28:
                    st.write("• Увеличьте прочность на 2 сутки")
                if 'Blaine' in input_values and input_values['Blaine'] < 4000:
                    st.write("• Повысьте тонкость помола")
                if 'Добавка' in input_values and input_values['Добавка'] > 12:
                    st.write(f"• Снизьте содержание добавки ({input_values['Добавка']:.1f}%)")
            else:
                st.success("✅ Параметры в оптимальном диапазоне!")

# Боковая панель
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 О модели")
    st.markdown(f"""
    - **Лист:** {selected_sheet}
    - **Образцов:** {len(X)}
    - **Признаков:** {len(feature_cols)}
    - **R²:** {r2:.3f}
    - **MAE:** {mae:.2f} МПа
    """)
    
    # Важность признаков
    if len(feature_cols) > 1:
        st.markdown("---")
        st.markdown("### 🔍 Важные параметры")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:5]
        for i, idx in enumerate(indices):
            st.progress(importances[idx], text=f"{feature_cols[idx]}: {importances[idx]:.2f}")
