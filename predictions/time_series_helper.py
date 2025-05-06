import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def to_3Darray(pd_X, pd_y, sequence_length):
    X, y = [], []
    for i in range(0,len(pd_X)-sequence_length):
        X.append(pd_X[i:i+sequence_length])
        y.append(pd_y[i+sequence_length])
    return np.array(X), np.array(y)


def make_data_for_model(calls_filename='../data/calls_by_district.csv', weather_filename='../data/weather_clean.csv', sequence_length=14):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    # Read and preprocess call data
    calls = pd.read_csv(calls_filename)
    calls['CallDateTime'] = pd.to_datetime(calls['CallDateTime'])
    calls.rename(columns={'NAME': 'location'}, inplace=True)
    calls['date'] = calls['CallDateTime'].dt.date
    calls['time_of_day'] = np.where((calls['CallDateTime'].dt.hour >= 8) & (calls['CallDateTime'].dt.hour < 20), 'Day', 'Night')

    # Aggregate calls per date/time_of_day/location
    grouped = calls.groupby(['date', 'time_of_day', 'location']).size().reset_index(name='num_calls')

    # Read and preprocess weather data
    weather = pd.read_csv(weather_filename)
    weather['date'] = pd.to_datetime(weather['DATE']).dt.date
    weather.drop(columns='DATE', inplace=True)

    # Merge calls and weather
    same_cols = grouped.columns.intersection(weather.columns).tolist()
    join = pd.merge(grouped, weather, on=same_cols, how='inner')

    # Time features
    join['date'] = pd.to_datetime(join['date'])
    join['day_of_week'] = join['date'].dt.dayofweek
    join['week_of_year'] = join['date'].dt.isocalendar().week.astype(int)
    join['month'] = join['date'].dt.month
    join['year'] = join['date'].dt.year
    join['day_of_year'] = join['date'].dt.dayofyear

    # Cyclical encodings
    join['sine_day_of_year'] = np.sin(2 * np.pi * join['day_of_year'] / 365.25)
    join['cosine_day_of_year'] = np.cos(2 * np.pi * join['day_of_year'] / 365.25)
    join['sine_day_of_week'] = np.sin(2 * np.pi * join['day_of_week'] / 7)
    join['cosine_day_of_week'] = np.cos(2 * np.pi * join['day_of_week'] / 7)
    join['sine_week_of_year'] = np.sin(2 * np.pi * join['week_of_year'] / 52.1775)
    join['cosine_week_of_year'] = np.cos(2 * np.pi * join['week_of_year'] / 52.1775)

    # Drop raw date-related columns
    join.drop(columns=['day_of_year', 'day_of_week', 'week_of_year', 'date'], inplace=True)

    # Sort and create lag features per group
    join = join.sort_values(['year', 'month', 'time_of_day', 'location']).copy()
    join['lag_1'] = join.groupby(['location', 'time_of_day'])['num_calls'].shift(1)
    join['lag_2'] = join.groupby(['location', 'time_of_day'])['num_calls'].shift(2)
    join['lag_7'] = join.groupby(['location', 'time_of_day'])['num_calls'].shift(7)
    join['rolling_3'] = join.groupby(['location', 'time_of_day'])['num_calls'].shift(1).rolling(window=3).mean().reset_index(0, drop=True)
    join['rolling_7'] = join.groupby(['location', 'time_of_day'])['num_calls'].shift(1).rolling(window=7).mean().reset_index(0, drop=True)

    # Drop missing rows from lag/rolling calc
    join = join.dropna().reset_index(drop=True)

    # Separate features and target
    pd_y = join['num_calls']
    pd_X = join.drop(columns=['num_calls'])

    # One-hot encode categoricals and scale numerics
    num_attribs = ['PRCP', 'SNOW', 'TMAX', 'TMIN', 'TOBS',
                   'sine_day_of_week', 'cosine_day_of_week',
                   'sine_week_of_year', 'cosine_week_of_year',
                   'sine_day_of_year', 'cosine_day_of_year',
                   'month', 'year',
                   'lag_1', 'lag_2', 'lag_7', 'rolling_3', 'rolling_7']
    cat_attribs = ['time_of_day', 'location']

    num_pipeline = Pipeline([('std_scaler', StandardScaler())])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs)
    ])

    X_prepared = full_pipeline.fit_transform(pd_X)

    # Convert to 3D array
    X, y = to_3Darray(X_prepared, pd_y, sequence_length=sequence_length)

    return X, y, len(X_prepared[0])