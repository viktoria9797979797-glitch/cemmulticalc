# cement_predictor_universal.py - ВЕРСИЯ 2.0 (работает с любыми листами)

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

def find_header_row(df, max_rows=30):
    """Находит строку, содержащую заголовки колонок"""
    keywords = ['sio2', 'cao', 'al2o3', 'fe2o3', 'прочн', 'блейн', 'blaine', 'SiO2', 'CaO']
    
    for i in range(min(max_rows, len(df))):
        row = df.iloc[i].astype(str).str.lower()
        row_text = ' '.join(row.values)
        
        # Проверяем, содержит ли строка ключевые слова
        for keyword in keywords:
            if keyword.lower() in row_text:
                # Также проверяем, что в строке есть хотя бы 3 непустых значения
                non_empty = sum(1 for val in row if val not in ['nan', 'None', ''])
                if non_empty >= 3:
                    return i
    return 0

def normalize_column_name(col_name):
    """Приводит название колонки к стандартному виду"""
    if pd.isna(col_name):
        return None
    
    col_str = str(col_name).strip().lower()
    col_clean = re.sub(r'[^\wа-я]', '', col_str)
    
    # Прочность
    if 'прочность2' in col_clean or '2сут' in col_clean or '2дн' in col_clean:
        return 'strength_2d'
    if 'прочность28' in col_clean or '28сут' in col_clean or '28дн' in col_clean:
        return 'strength_28d'
    
    # Химический состав
    if col_clean in ['sio2', 'кремнезём', 'кремнезем', 'si02']:
        return 'SiO2'
    if col_clean in ['al2o3', 'al2oз', 'глинозём', 'глинозем', 'al203']:
        return 'Al2O3'
    if col_clean in ['fe2o3', 'fe2oз', 'оксиджелеза', 'fe203']:
        return 'Fe2O3'
    if col_clean in ['cao', 'оксидкальция']:
        return 'CaO'
    if col_clean in ['mgo', 'оксидмагния']:
        return 'MgO'
    if col_clean in ['so3', 'сульфаты']:
        return 'SO3'
    if 'na2o' in col_clean:
        return 'Na2OEQ'
    if col_clean in ['ппп', 'потери']:
        return 'ППП'
    if 'caoсв' in col_clean or 'cao_free' in col_clean or 'caosv' in col_clean:
        return 'CaO_free'
    if col_clean in ['но']:
        return 'НО'
    
    # Физические характеристики
    if 'блейн' in col_clean or 'blaine' in col_clean:
        return 'Blaine'
    if '45мкм' in col_clean or '45мк' in col_clean or 'sieve45' in col_clean:
        return 'sieve_45'
    if col_clean in ['нг']:
        return 'НГ'
    if col_clean in ['рио']:
        return 'РИО'
    
    # Сроки схватывания
    if 'начало' in col_clean:
        return 'start_set'
    if 'конец' in col_clean:
        return 'end_set'
    
    # Добавки
    if 'известняк' in col_clean:
        return 'Добавка'
    if 'шлак' in col_clean:
        return 'Добавка'
    if 'клинкер' in col_clean:
        return 'Клинкер'
    
    return None

@st.cache_data
def load_sheet_data(sheet_name):
    """Загружает данные из выбранного листа с любой структурой"""
    try:
        # Читаем весь лист без заголовков
        df_raw = pd.read_excel(DATA_FILE, sheet_name=sheet_name, header=None)
        
        # Находим строку с заголовками
        header_row = find_header_row(df_raw)
        
        if header_row > 0:
            # Пропускаем строки до заголовка
            df = pd.read_excel(DATA_FILE, sheet_name=sheet_name, skiprows=header_row)
        else:
            df = df_raw.copy()
        
        # Нормализуем названия колонок
        new_columns = {}
        for col in df.columns:
            new_name = normalize_column_name(col)
            if new_name:
                new_columns[col] = new_name
        
        # Если не нашли стандартных названий, пробуем первую строку как заголовок
        if len(new_columns) < 3:
            # Берём первую непустую строку как заголовок
            for i in range(min(5, len(df_raw))):
                row = df_raw.iloc[i].astype(str)
                if any('проч' in str(x).lower() for x in row if pd.notna(x)):
                    df = pd.read_excel(DATA_FILE, sheet_name=sheet_name, skiprows=i)
                    break
            
            # Снова нормализуем
            for col in df.columns:
                new_name = normalize_column_name(col)
                if new_name:
                    new_columns[col] = new_name
        
        df = df.rename(columns=new_columns)
        
        # Преобразуем в числа
        for col in df.columns:
            if col not in ['Тип_цемента', 'Клинкер']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Удаляем строки, где все значения NaN
        df = df.dropna(how='all')
        
        # Если есть колонка Клинкер, используем её для определения добавки
        if 'Клинкер' in df.columns:
            # Значения клинкера обычно 95-96%, тогда добавка = 100 - клинкер
            df['Добавка'] = 100 - df['Клинкер']
        
        # Удаляем строки с пропусками в целевой переменной
        if 'strength_28d' in df.columns:
            df = df.dropna(subset=['strength_28d'])
        
        return df
    except Exception as e:
        st.error(f"Ошибка загрузки {sheet_name}: {e}")
        return None

@st.cache_data
def get_all_sheets():
    """Получает список всех листов в Excel-файле"""
    try:
        xl = pd.ExcelFile(DATA_FILE)
        return xl.sheet_names
    except:
        return []

# Получаем список листов
sheets = get_all_sheets()

if not sheets:
    st.error(f"❌ Файл {DATA_FILE} не найден!")
    st.info("Убедитесь, что файл загружен в репозиторий")
    st.stop()

# Выбор листа
st.sidebar.header("📁 Выбор данных")
selected_sheet = st.sidebar.selectbox("Выберите лист с данными", sheets)

# Загрузка данных
df = load_sheet_data(selected_sheet)

if df is None or len(df) < 3:
    st.error(f"❌ Недостаточно данных в листе '{selected_sheet}' (найдено {len(df) if df is not None else 0} строк)")
    
    # Показываем диагностику
    with st.expander("🔍 Диагностика: что видит программа"):
        try:
            df_raw = pd.read_excel(DATA_FILE, sheet_name=selected_sheet, header=None)
            st.write("**Первые 10 строк файла:**")
            st.dataframe(df_raw.head(10))
            st.write("**Имена колонок после обработки:**")
            if df is not None:
                st.write(list(df.columns))
        except Exception as e:
            st.write(f"Ошибка диагностики: {e}")
    st.stop()

# Показываем информацию о данных
st.subheader(f"📊 Данные из листа: {selected_sheet}")
st.write(f"**Найдено образцов:** {len(df)}")

# Определяем доступные колонки
available_cols = [col for col in df.columns if col not in ['Клинкер', 'Тип_цемента']]
available_cols = [col for col in available_cols if df[col].notna().any()]

st.write("**Доступные параметры:**", ", ".join(available_cols))

# Проверяем наличие целевой переменной
if 'strength_28d' not in df.columns:
    st.error("❌ Не найдена колонка с прочностью 28 суток!")
    st.info("Убедитесь, что в таблице есть колонка с названием 'Прочность 28 сут.' или '28 сут'")
    st.stop()

if 'strength_2d' not in df.columns:
    st.warning("⚠️ Нет данных о прочности на 2 сутки. Будут использованы только химические параметры.")

# Подготовка признаков
feature_cols = [col for col in available_cols if col != 'strength_28d']

if len(feature_cols) == 0:
    st.error("❌ Нет доступных признаков для обучения модели!")
    st.stop()

X = df[feature_cols].copy()
y = df['strength_28d'].copy()

# Удаляем строки с пропусками
valid = y.notna()
for col in X.columns:
    valid = valid & X[col].notna()

X = X[valid]
y = y[valid]

if len(X) < 3:
    st.error(f"❌ После очистки осталось только {len(X)} строк. Нужно минимум 3.")
    st.stop()

# Обучение модели
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Для тестовой выборки может быть всего 1 элемент
    if len(X_test) >= 1:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_train_scaled[:1]
        y_test = y_train[:1]
    
    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 0
    mae = mean_absolute_error(y_test, y_pred) if len(y_test) > 1 else 0
    
    # Отображение метрик
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{r2:.3f}" if len(y_test) > 1 else "N/A")
    col2.metric("MAE", f"{mae:.2f} МПа" if len(y_test) > 1 else "N/A")
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
        available_chem = [c for c in chem_cols if c in feature_cols and df[c].notna().any()]
        
        if available_chem:
            cols = st.columns(min(3, len(available_chem)))
            for i, col_name in enumerate(available_chem):
                with cols[i % 3]:
                    min_val = float(df[col_name].min())
                    max_val = float(df[col_name].max())
                    default_val = float(df[col_name].median())
                    input_values[col_name] = st.slider(col_name, min_val, max_val, default_val, 0.1, format="%.2f")
        
        st.markdown("### ⚙️ Физические характеристики")
        
        phys_cols = ['Blaine', 'sieve_45', 'НГ', 'РИО']
        available_phys = [c for c in phys_cols if c in feature_cols and df[c].notna().any()]
        
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
        
        st.markdown("### ⏱️ Сроки схватывания")
        
        time_cols = ['start_set', 'end_set']
        available_time = [c for c in time_cols if c in feature_cols and df[c].notna().any()]
        
        if available_time:
            cols = st.columns(2)
            for i, col_name in enumerate(available_time):
                with cols[i % 2]:
                    min_val = float(df[col_name].min())
                    max_val = float(df[col_name].max())
                    default_val = float(df[col_name].median())
                    input_values[col_name] = st.slider(col_name, min_val, max_val, default_val, 5.0, format="%.0f")
        
        # Прочность 2 суток
        if 'strength_2d' in feature_cols and df['strength_2d'].notna().any():
            st.markdown("### 💪 Прочность на ранних сроках")
            min_val = float(df['strength_2d'].min())
            max_val = float(df['strength_2d'].max())
            default_val = float(df['strength_2d'].median())
            input_values['strength_2d'] = st.slider("Прочность 2 суток (МПа)", min_val, max_val, default_val, 0.5, format="%.1f")
        
        # Добавка
        if 'Добавка' in feature_cols and df['Добавка'].notna().any():
            st.markdown("### 🧱 Добавки")
            min_val = float(df['Добавка'].min())
            max_val = float(df['Добавка'].max())
            default_val = float(df['Добавка'].median())
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

except Exception as e:
    st.error(f"Ошибка при обучении модели: {e}")

# Боковая панель
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 О модели")
    st.markdown(f"""
    - **Лист:** {selected_sheet}
    - **Образцов:** {len(X) if 'X' in locals() else 0}
    - **Признаков:** {len(feature_cols) if 'feature_cols' in locals() else 0}
    """)
    
    if 'model' in locals() and len(feature_cols) > 1:
        st.markdown("---")
        st.markdown("### 🔍 Важные параметры")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:5]
        for i, idx in enumerate(indices):
            st.progress(importances[idx], text=f"{feature_cols[idx]}: {importances[idx]:.2f}")
