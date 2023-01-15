import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# Функция по предобработке данных: приведение к нужному формату, удаление из строки единицы измерения, замена значений
csv_path = '/Users/eshishmarina/Downloads/nba2k-full.csv'
def clean_data(path):
    df = pd.read_csv(path)
    df['b_day'] = pd.to_datetime(df['b_day'])
    df['draft_year'] = pd.to_datetime(df['draft_year'], format='%Y')
    df['team'] = df['team'].fillna('No Team')
    df[['ft_height', 'height']] = df['height'].str.split('/',expand=True)
    df['height'] = df['height'].astype(float)
    df[['lbs_weight', 'weight']] = df['weight'].str.split('/',expand=True)
    df['weight'] = df['weight'].str.replace(' kg.','').astype(float)
    df['salary'] = df['salary'].str.replace('$','').astype(float)
    df['draft_round'] = df['draft_round'].str.replace('Undrafted','0')
    df['country'] = df['country'].where((df.country == 'USA'), other='Not-USA')
    df = df.drop(columns=['ft_height', 'lbs_weight'])
    return df

df_cleaned = clean_data(csv_path)

# Функция продолжает предобработку данных, в т.ч. убираем колонки, где кол-во уникальных значений больше 50, чтобы не утяжелять ML-модель
def feature_data(df_cleaned):
    df_cleaned['version'] = df_cleaned['version'].str.replace('NBA', '')
    df_cleaned['version'] = df_cleaned['version'].str.replace('k', '0')
    df_cleaned['version'] = pd.to_datetime(df_cleaned['version'], format='%Y')
    df_cleaned['age'] = np.ceil((df_cleaned['version'] - df_cleaned['b_day'])/ np.timedelta64(1, 'Y')).astype(int)
    df_cleaned['experience'] = ((df_cleaned['version'] - df_cleaned['draft_year'])/ np.timedelta64(1, 'Y')).round(0).astype(int)
    df_cleaned['bmi'] = df_cleaned['weight'] / (df_cleaned['height'])**2
    df_cleaned = df_cleaned.drop(columns=['version', 'b_day', 'draft_year', 'weight', 'height'])

    df_type = pd.DataFrame(df_cleaned.dtypes.rename('object')).reset_index()
    df_unique = pd.DataFrame(df_cleaned.nunique().rename('number_unique').reset_index())
    a = df_unique.merge(df_type)
    a['object'] = a['object'].astype(str)
    b = a[a.object == 'object']
    b = b[b.number_unique > 50]
    cols_to_drop = b['index'].to_list()
    df = df_cleaned.drop(columns=cols_to_drop)
    return df


df_featured = feature_data(df_cleaned)

# Функция проверяет признаки на мультиколлинеарность, чтобы признаки не имели друг с другом линейной зависимости
def multicol_data(df_featured):
    corr_map_features = df_featured[['rating', 'age', 'experience', 'bmi']]
    corr_map = corr_map_features.corr().round(2)
    corr_map_1 = corr_map.stack()
    print(corr_map_1[(corr_map_1 > 0.5) & (corr_map_1 != 1) & (corr_map_1 < -0.5)])
    # выявлена высокая корреляция между возрастом и опытом, считаем корреляцию этих параметров с зарплатой
    r_age = df_featured['age'].corr(df_featured['salary'])
    r_experience = df_featured['experience'].corr(df_featured['salary'])
    # колонку возраста удаляем, т.к. его корреляция с зарплатой выше, чем опыта с зарплатой
    df = df_featured.drop(columns='age')
    return df

df = multicol_data(df_featured)

# Функция нормализует отдельно числовые и категориальные переменные (столбцы), чтобы они корректно потом зашли в ML-модель
def transform_data(df):
    num_feat_df = df.select_dtypes('number')  # numerical features
    cat_feat_df = df.select_dtypes('object')  # categorical features

    # нормализуем числовые параметры
    scaler = StandardScaler()
    scaler.fit(df[['rating', 'experience', 'bmi']])
    scaler_result = scaler.transform(df[['rating', 'experience', 'bmi']])
    scaler_df = pd.DataFrame(scaler_result)
    scaler_df.columns = ['rating', 'experience', 'bmi']

    # нормализуем категориальные переменные
    enc = OneHotEncoder(sparse=False)
    encoder_result = enc.fit_transform(cat_feat_df)
    encoder_df = pd.DataFrame(encoder_result)
    columns = []
    for cat in enc.categories_:
        for el in cat:
            columns.append(el)
    encoder_df.columns = columns

    all_new_features = pd.concat((scaler_df, encoder_df), axis=1)

    X = all_new_features
    y = df['salary']
    return X, y

X, y = transform_data(df)

answer = {
    'shape': [X.shape, y.shape],
    'features': list(X.columns),
    }


