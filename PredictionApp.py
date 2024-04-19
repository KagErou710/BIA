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
Receipts = base64.b64encode(open("picture/Receipts.png", "rb").read()).decode("utf-8")
Receipts = f"data:image/png;base64,{Receipts}"
Show = base64.b64encode(open("picture/Show.png", "rb").read()).decode("utf-8")
Show = f"data:image/png;base64,{Show}"

dashboard = base64.b64encode(open("picture/Dashboard.png", "rb").read()).decode("utf-8")
dashboard = f"data:image/png;base64,{dashboard}"



dfQuart = pd.read_csv('Data3/NoOfArrivals.csv')
dfStd = pd.read_csv('Data3/NoOfArrivals2CvdDeleted.csv')
dfMonth = pd.read_csv('Data3/NoOfArrivals3Foreign.csv')
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
nationalities_dict2 = contentsDict['Arrival2']['Country'][1]


nationalities = list(nationalities_dict.keys())
nationalities2 = list(nationalities_dict2.keys())
# isCovid = list(covid_dict.keys())

stay_days_range = range(1, 31)

app = dash.Dash(__name__)
style={'display': 'flex', 'flex-direction': 'row', 'height': '400px'}

app.layout = html.Div(children=[

    html.H1('DSS Thailand Tourism Industry', style={'background-color': '#e6e6ff'}),
    html.Div(style={'display': 'flex', 'flex-direction': 'row', 'height': '400px'}, children=[
    html.Div(style={'width': '30%'}, children=[
    html.Div(children=[
        html.Img(style={'width': '30%', 'height': '30%'}, src=nationality, alt="Example Image"),
        dcc.Dropdown(
            id='nationality-dropdown',
            options=[{'label': r, 'value': r} for r in nationalities],
            value=nationalities[0],
        ),
    ]),  
    html.Div(children=[
        html.Img(style={'width': '10%', 'height': '10%'}, src=year, alt="Example Image"),
        html.P(''),
        dcc.Input(id='year', type='number', value='2025')
    ]),
    html.Div(children=[
        html.Img(style={'width': '20%', 'height': '20%'}, src=Receipts, alt="Example Image"),
        html.P(''),
        dcc.Input(id='receipt', type='number', value=None)
    ]),
    html.Button(
                children=[
            html.Img(src=Show, className="button-image")
        ],
        className="button-container",
        id="button",),
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
        Input('nationality-dropdown', 'value'),
        Input('receipt', 'value'),
        Input('year', 'value'),

    ]
)
def update_output(
    n_clicks,
    selected_nationality,
    selected_receipt,
    selected_year,
):
    model1 = joblib.load('Model1.joblib')
    model2 = joblib.load('Model2.joblib')
    model3 = joblib.load('ReceiptsModel.joblib')
    nationality = contentsDict['Arrival']['Country'][1][selected_nationality]
    nationality2 = contentsDict['Arrival2']['Country'][1][selected_nationality]
    # selected_region = contentsDict['Arrival']['Region'][1][CountryToRegion[selected_nationality]]
    if n_clicks is None:
        return []
    try:
        Receipts = selected_receipt
        if Receipts == None:
            print('Receipt Model')
            Receipts = predReceipt(
                model3,
                nationality,
                selected_year
                )[0]
        pred1 = predArr(
            model1,
            2,
            nationality,
            selected_year,
            Receipts
            )

        pred2 = predArr(
            model2,
            2,
            nationality2,
            selected_year,
            Receipts
        )
        pred_y = ensemble_learning(
            pred1,
            pred2
            )
        return[
            html.Img(src=output, alt="Example Image"),
            html.P('Preds: {}'.format(pred_y))
        ]
            
            
    except:
        return [html.P('something is missed')]

def predArr(Model, degree, Country, Year, Receipts):
    # numecallize input
    inputs = [
        Country,
        Year,
        Receipts
              ]
    x_pred = np.array([inputs])
    poly = PolynomialFeatures(degree=degree)
    x_pred = poly.fit_transform(x_pred)
    return(Model.predict(x_pred))


def predReceipt(Model, Country, Year):
    # numecallize input
    inputs = [
        Country,
        Year
              ]
    x_pred = np.array([inputs])
    poly = PolynomialFeatures(degree=3)
    x_pred = poly.fit_transform(x_pred)
    return(Model.predict(x_pred))



def ensemble_learning(a, b):
    weights = [0.5109439720318428, 0.4890560279681572]
    y_pred = np.dot(weights, [a, b])
    return y_pred

if __name__ == '__main__':
    app.run_server(debug=True)
