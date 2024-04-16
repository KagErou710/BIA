import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import base64

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
import joblib
from sklearn.preprocessing import PolynomialFeatures


nationality = base64.b64encode(open("picture/Nationality.png", "rb").read()).decode("utf-8")
nationality = f"data:image/png;base64,{nationality}"
output = base64.b64encode(open("picture/Output.png", "rb").read()).decode("utf-8")
output = f"data:image/png;base64,{output}"
year = base64.b64encode(open("picture/Year.png", "rb").read()).decode("utf-8")
year = f"data:image/png;base64,{year}"
GDP = base64.b64encode(open("picture/GDP.png", "rb").read()).decode("utf-8")
GDP = f"data:image/png;base64,{GDP}"
Inflation = base64.b64encode(open("picture/Inflation.png", "rb").read()).decode("utf-8")
Inflation = f"data:image/png;base64,{Inflation}"
Receipts = base64.b64encode(open("picture/Receipts.png", "rb").read()).decode("utf-8")
Receipts = f"data:image/png;base64,{Receipts}"
Spending = base64.b64encode(open("picture/Spending.png", "rb").read()).decode("utf-8")
Spending = f"data:image/png;base64,{Spending}"
StayDays = base64.b64encode(open("picture/StayDays.png", "rb").read()).decode("utf-8")
StayDays = f"data:image/png;base64,{StayDays}"

dashboard = base64.b64encode(open("picture/Dashboard.png", "rb").read()).decode("utf-8")
dashboard = f"data:image/png;base64,{dashboard}"
dashboard2 = base64.b64encode(open("picture/Dashboard2.png", "rb").read()).decode("utf-8")
dashboard2 = f"data:image/png;base64,{dashboard2}"



dfQuart = pd.read_csv('Data3/NoOfArrivals.csv')
dfStd = pd.read_csv('Data3/NoOfArrivals2CvdDeleted.csv')
dfList = [dfQuart, dfStd]
nameList = ['Arrival', 'Arrival2']

CountryToRegion = {}
for i, row in dfQuart.iterrows():
    if row['Country'] not in CountryToRegion:
        CountryToRegion[row['Country']] = row['Region']

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
# covid_dict = contentsDict['Arrival2']['Covid Outbreak'][1]

nationalities = list(nationalities_dict.keys())
nationalities2 = list(nationalities_dict2.keys())
regions = list(region_dict.keys())
# isCovid = list(covid_dict.keys())

stay_days_range = range(1, 31)


years = []
for i in range(2023, 2101):
    years.append(i)

app = dash.Dash(__name__)
style={'display': 'flex', 'flex-direction': 'row', 'height': '400px'}

app.layout = html.Div(children=[

    html.H1('アンケートフォーム', style={'background-color': '#e6e6ff'}),
    html.Div(style={'display': 'flex', 'flex-direction': 'row', 'height': '400px'}, children=[
    html.Div(style={'width': '30%'}, children=[
    html.Div(children=[
        html.Img(src=nationality, alt="Example Image"),
        dcc.Dropdown(
            id='nationality-dropdown',
            options=[{'label': r, 'value': r} for r in nationalities],
            value=nationalities[0],
        ),
    ]),
    # html.Div(children=[
    #     html.Label('Region:'),
    #     dcc.Dropdown(
    #         id='region-dropdown',
    #         options=[{'label': r, 'value': region_dict[r]} for r in regions],
    #         value=region_dict[regions[0]]
    #     ),
    # ]),    
    html.Div(children=[
        html.Img(src=StayDays, alt="Example Image"),
        html.P(''),
        dcc.Dropdown(
            id='stay-days-dropdown',
            options=[{'label': str(d), 'value': d} for d in stay_days_range],
            value=stay_days_range[0],

        ),
    ]),
    # html.Div(children=[
    #     html.Label('Covid:'),
    #     dcc.Dropdown(
    #         id='covid-dropdown',
    #         options=[{'label': str(d), 'value': covid_dict[d]} for d in isCovid],
    #         value=covid_dict[isCovid[0]]
    #     ),
    # ]),
    html.Div(children=[
        html.Img(style={'width': '20%', 'height': '20%'}, src=Spending, alt="Example Image"),
        html.P(''),
        dcc.Input(id='spending', type='number', value='初期値')
    ]),
    html.Div(children=[
        html.Img(src=Receipts, alt="Example Image"),
        html.P(''),
        dcc.Input(id='receipt', type='number', value='初期値')
    ]),
    html.Div(children=[
        html.Img(src=year, alt="Example Image"),
        html.P(''),
        dcc.Input(id='year', type='number', value='初期値')
    ]),
    html.Div(children=[
        html.Img(src=Inflation, alt="Example Image"),
        html.P(''),
        dcc.Input(id='inflation', type='number', value='初期値')
    ]),
    html.Div(children=[
        html.Img(src=GDP, alt="Example Image"),
        html.P(''),
        dcc.Input(id='gdp', type='number', value='初期値')
    ]),
    html.Button('表示', id='button'),
    html.Div(id='output-container', children=[])
    ]),
    html.Div(children=[
        html.Img(src=dashboard, alt="Example Image"),
        # html.Img(src=dashboard2, alt="Example Image")
    ])
    ])
    ]


)


@app.callback(
    Output('output-container', 'children'),
    [
        Input('button', 'n_clicks'),
        # Input('region-dropdown', 'value'),
        Input('nationality-dropdown', 'value'),
        Input('stay-days-dropdown', 'value'),
        # Input('covid-dropdown', 'value'),
        Input('spending', 'value'),
        Input('receipt', 'value'),
        Input('year', 'value'),
        Input('inflation', 'value'),
        Input('gdp', 'value')
    ]
)
def update_output(
    n_clicks,
    # selected_region,
    selected_nationality,
    selected_staydays,
    # selected_covid,
    selected_spending,
    selected_receipt,
    selected_year,
    selected_inflation,
    selected_gdp
):
    model1 = joblib.load('Model1.joblib')
    model2 = joblib.load('Model2.joblib')
    print('a')
    print(contentsDict['Arrival']['Country'][1][selected_nationality])
    print('b')
    # print(CountryToRegion)
    print(CountryToRegion[selected_nationality])
    nationality = contentsDict['Arrival']['Country'][1][selected_nationality]
    selected_region = contentsDict['Arrival']['Region'][1][CountryToRegion[selected_nationality]]
    if n_clicks is None:
        return []
# try:
    pred1 = Pred1(
        model1,
        nationality,
        selected_staydays,
        selected_spending,
        selected_receipt,
        selected_region,
        selected_year)

    pred2 = Pred2(
        model2,
        nationality,
        selected_inflation,
        selected_year,
        selected_gdp,
        selected_staydays,
        selected_spending,
        selected_receipt,
        selected_region,
        # selected_covid
    )
    pred_y = ensemble_learning(
        pred1,
        pred2
        )
    return[
        html.Img(src=output, alt="Example Image"),
        html.P('Preds: {}'.format(pred_y))
    ]

    # except:
    #     print(selected_nationality,
    #         selected_inflation,
    #         selected_year,
    #         selected_gdp,
    #         selected_staydays,
    #         selected_spending,
    #         selected_receipt,
    #         selected_region,)
    #     return [html.P('something is missed')]

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
    # x_pred = np.format_float_positional(x_pred, precision=7, fractional=False, exponential=True)
    # print(x_pred)
    poly = PolynomialFeatures(degree=2)
    x_pred = poly.fit_transform(x_pred)
    return(Model.predict(x_pred))


def Pred2(Model, Country, Inflation, Year, GDP, StayDays, Spending , Receipts, Region):
    # numecallize input
    inputs = [
        Country,
        Inflation,
        Year,
        GDP,
        StayDays,
        Spending,
        Receipts,
        Region
              ]
    x_pred = np.array([inputs])
    poly = PolynomialFeatures(degree=2)
    x_pred = poly.fit_transform(x_pred)
    return(Model.predict(x_pred))


def ensemble_learning(a, b):
    weights = [0.4995045442723615, 0.5004954557276384]
    y_pred = np.dot(weights, [a, b])
    return y_pred


# アプリケーションの実行
if __name__ == '__main__':
    app.run_server(debug=True)
