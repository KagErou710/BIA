import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
import joblib
from sklearn.preprocessing import PolynomialFeatures


dfQuart = pd.read_csv('Data2/TourismArrival2020-2021(Quarterly).csv')
dfStd = pd.read_csv('Data3/NoOfArrivals2.csv')
dfList = [dfQuart, dfStd]
nameList = ['Arrival', 'Arrival2']

contentsDict = {}
for i, df in enumerate(dfList):
    name = nameList[i]
    columns = df.columns.tolist()
    temp2 = {}
    for column in columns:
        if df[column].dtype != object:
            continue
        temp = df[column].unique()
        AnyToNum = {}
        NumToAny = {}
        for j, item in enumerate(temp, start=1):
            NumToAny[j] = item
            AnyToNum[item] = j
        def converter(item):
            return AnyToNum.get(item, np.nan)
        df[column] = df[column].apply(converter)
        
        temp2[column] = [NumToAny, AnyToNum]
    contentsDict[name] = temp2


nationalities_dict = contentsDict['Arrival']['Country'][1]
region_dict = contentsDict['Arrival']['Region'][1]

nationalities_dict2 = contentsDict['Arrival2']['Country'][1]
region_dict = contentsDict['Arrival2']['Region'][1]
covid_dict = contentsDict['Arrival2']['Covid Outbreak'][1]

nationalities = list(nationalities_dict.keys())
nationalities2 = list(nationalities_dict2.keys())
regions = list(region_dict.keys())
isCovid = list(covid_dict.keys())
print(isCovid)
print(regions)

stay_days_range = range(1, 31)


years = []
for i in range(2023, 2101):
    years.append(i)






app = dash.Dash(__name__)


app.layout = html.Div(children=[
    html.H1('アンケートフォーム'),
    html.Div(children=[
        html.Label('Country:'),
        dcc.Dropdown(
            id='nationality-dropdown',
            options=[{'label': r, 'value': nationalities_dict2[r]} for r in nationalities2],
            value=nationalities_dict2[nationalities2[0]]
        ),
    ]),
    html.Div(children=[
        html.Label('Region:'),
        dcc.Dropdown(
            id='region-dropdown',
            options=[{'label': r, 'value': region_dict[r]} for r in regions],
            value=region_dict[regions[0]]
        ),
    ]),    
    html.Div(children=[
        html.Label('Stay days:'),
        dcc.Dropdown(
            id='stay-days-dropdown',
            options=[{'label': str(d), 'value': d} for d in stay_days_range],
            value=stay_days_range[0]
        ),
    ]),
    html.Div(children=[
        html.Label('Covid:'),
        dcc.Dropdown(
            id='covid-dropdown',
            options=[{'label': str(d), 'value': covid_dict[d]} for d in isCovid],
            value=covid_dict[isCovid[0]]
        ),
    ]),
    html.Div(children=[
        html.H3('Spending:'),
        dcc.Input(id='spending', type='number', value='初期値')
    ]),
    html.Div(children=[
        html.H3('Receipt:'),
        dcc.Input(id='receipt', type='number', value='初期値')
    ]),
    html.Div(children=[
        html.H3('Year:'),
        dcc.Input(id='year', type='number', value='初期値')
    ]),
    html.Div(children=[
        html.H3('Inflation Rate:'),
        dcc.Input(id='inflation', type='number', value='初期値')
    ]),
    html.Div(children=[
        html.H3('GDP:'),
        dcc.Input(id='gdp', type='number', value='初期値')
    ]),
    html.Button('表示', id='button'),
    html.Div(id='output-container', children=[])
])


@app.callback(
    Output('output-container', 'children'),
    [
        Input('button', 'n_clicks'),
        Input('region-dropdown', 'value'),
        Input('nationality-dropdown', 'value'),
        Input('stay-days-dropdown', 'value'),
        Input('covid-dropdown', 'value'),
        Input('spending', 'value'),
        Input('receipt', 'value'),
        Input('year', 'value'),
        Input('inflation', 'value'),
        Input('gdp', 'value')
    ]
)
def update_output(
    n_clicks,
    selected_region,
    selected_nationality,
    selected_staydays,
    selected_covid,
    selected_spending,
    selected_receipt,
    selected_year,
    selected_inflation,
    selected_gdp
):
    model1 = joblib.load('Model1.joblib')
    model2 = joblib.load('Model2.joblib')
    if n_clicks is None:
        return []
    try:
        print(selected_nationality)
        print(selected_covid)
        pred1 = Pred1(
            model1,
            selected_nationality,
            selected_staydays,
            selected_spending,
            selected_receipt,
            selected_region,
            selected_year)

        pred2 = Pred2(
            model2,
            selected_nationality,
            selected_inflation,
            selected_year,
            selected_gdp,
            selected_staydays,
            selected_spending,
            selected_receipt,
            selected_region,
            selected_covid
        )
        pred_y = ensemble_learning(
            pred1,
            pred2
            )
        return[
            html.P('Preds: {}'.format(pred_y)),
        ]

    except:
        print('something is missed')
        return []

def Pred1(Model, Country, StayDays, Spending, Receipts, Region, Year):
    # numecallize input
    inputs = [
        Country,
        StayDays,
        Spending,
        Receipts,
        Region,
        Year
              ]
    x_pred = np.array([inputs])
    poly = PolynomialFeatures(degree=3)
    x_pred = poly.fit_transform(x_pred)
    return(Model.predict(x_pred))


def Pred2(Model, Country, Inflation, Year, GDP, StayDays, Spending , Receipts, Region, Covid):
    # numecallize input
    inputs = [
        Country,
        Inflation,
        Year,
        GDP,
        StayDays,
        Spending,
        Receipts,
        Region,
        Covid
              ]
    x_pred = np.array([inputs])
    poly = PolynomialFeatures(degree=3)
    x_pred = poly.fit_transform(x_pred)
    return(Model.predict(x_pred))


def ensemble_learning(a, b):
    weights = [0.4995045442723615, 0.5004954557276384]
    y_pred = np.dot(weights, [a, b])
    return y_pred


# アプリケーションの実行
if __name__ == '__main__':
    app.run_server(debug=True)
