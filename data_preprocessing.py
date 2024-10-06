import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the dataset
def load_and_preprocess_data():
    df = pd.read_csv('AmesHousing.csv')

    features = ['Yr Sold', 'SalePrice', 'Lot Area', 'Overall Qual', 'Year Built', 'Gr Liv Area',
                'Total Bsmt SF', 'Garage Cars', 'Full Bath', 'Bedroom AbvGr', 'Kitchen Qual', 
                'Fireplaces', 'Garage Area', 'Neighborhood']

    df = df[features]
    df = pd.get_dummies(df, columns=['Neighborhood', 'Kitchen Qual'], drop_first=True)
    df = df.sort_values('Yr Sold')
    df = df.dropna()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df.drop(columns=['Yr Sold', 'SalePrice']))

    scaled_data = pd.DataFrame(scaled_data, columns=df.columns.drop(['Yr Sold', 'SalePrice']))
    scaled_data['SalePrice'] = scaler.fit_transform(df['SalePrice'].values.reshape(-1, 1))

    return scaled_data, scaler
