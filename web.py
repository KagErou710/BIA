import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings


dfAge = pd.read_csv('Data2/By Age 2015 - 2020.csv', dtype={'No of Arrivals': 'int', 'Year': 'int'})
dfIncome = pd.read_csv('Data2/By Annual Income Group(2019-2020).csv', dtype={'No of Arrival per Annual Income': 'int', 'Year': 'int'})
dfFreq = pd.read_csv('Data2/By Frequency of purpose(2015- 2020).csv', dtype={'No of Arrival by Frequency of purpose': 'int', 'Year': 'int'})
dfTransport = pd.read_csv('Data2/By mode of transport(2015-2020).csv')
dfOccupation = pd.read_csv('Data2/By Occupation(2015-2020).csv')
dfSex = pd.read_csv('Data2/By Sex 2015 - 2020.csv')
dfQuart = pd.read_csv('Data2/TourismArrival2020-2021(Quarterly).csv')
dfList = [dfAge, dfIncome, dfFreq, dfTransport, dfOccupation, dfSex, dfQuart]
nameList = ['Age', 'Income', 'Freq', 'Transport', 'Occupation', 'Sex', 'Arrival']

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




# データの準備
# nationalities_dict = {'日本': 0, 'アメリカ': 1, '中国': 2, 'イギリス': 3, 'フランス': 4}
nationalities_dict = contentsDict['Age']['Country'][1]
region_dict = contentsDict['Age']['Region'][1]
income_dict =  contentsDict['Income']['Income'][1]
frequency_dict  = contentsDict['Freq']['Frequency'][1]
transport_dict  = contentsDict['Transport']['Transport'][1]
occupation_dict  = contentsDict['Occupation']['Occupation'][1]
sex_dict = contentsDict['Sex']['Sex'][1]

nationalities = list(nationalities_dict.keys())
regions = list(region_dict.keys())
incomes = list(income_dict.keys())
frequencies = list(frequency_dict.keys())
transports = list(transport_dict.keys())
occupations = list(occupation_dict.keys())
sexes = list(sex_dict.keys())
stay_days_range = range(1, 31)


years = []
for i in range(2023, 2101):
    years.append(i)

# アプリケーションの定義
app = dash.Dash(__name__)

# レイアウトの定義
app.layout = html.Div(children=[
    html.H1('アンケートフォーム'),
    html.Div(children=[
        html.Label('地域:'),
        dcc.Dropdown(
            id='region-dropdown',
            options=[{'label': r, 'value': region_dict[r]} for r in regions],
            value=region_dict[regions[0]]
        ),
    ]),
    html.Div(children=[
        html.Label('所得:'),
        dcc.Dropdown(
            id='income-dropdown',
            options=[{'label': i, 'value': income_dict[i]} for i in incomes],
            value=income_dict[incomes[0]]
        ),
    ]),
    html.Div(children=[
        html.Label('訪問回数:'),
        dcc.Dropdown(
            id='frequency-dropdown',
            options=[{'label': f, 'value': frequency_dict[f]} for f in frequencies],
            value=frequency_dict[frequencies[0]]
        ),
    ]),
    html.Div(children=[
        html.Label('交通手段:'),
        dcc.Dropdown(
            id='transport-dropdown',
            options=[{'label': t, 'value': transport_dict[t]} for t in transports],
            value=transport_dict[transports[0]]
        ),
    ]),
    html.Div(children=[
        html.Label('職業:'),
        dcc.Dropdown(
            id='occupation-dropdown',
            options=[{'label': o, 'value': occupation_dict[o]} for o in occupations],
            value=occupation_dict[occupations[0]]
        ),
    ]),
    html.Div(children=[
        html.Label('性別:'),
        dcc.Dropdown(
            id='sex-dropdown',
            options=[{'label': s, 'value': sex_dict[s]} for s in sexes],
            value=sex_dict[sexes[0]]
        ),
    ]),
    html.Div(children=[
        html.Label('滞在日数:'),
        dcc.Dropdown(
            id='stay-days-dropdown',
            options=[{'label': str(d), 'value': d} for d in stay_days_range],
            value=stay_days_range[0]
        ),
    ]),
    html.Div(id='output-container')
])

# コールバック関数の定義
@app.callback(
    Output('output-container', 'children'),
    [
        Input('region-dropdown', 'value'),
        Input('income-dropdown', 'value'),
        Input('frequency-dropdown', 'value'),
        Input('transport-dropdown', 'value'),
        Input('occupation-dropdown', 'value'),
        Input('sex-dropdown', 'value'),
        Input('stay-days-dropdown', 'value')
    ]
)
def update_output(
    selected_region,
    selected_income,
    selected_frequency,
    selected_transport,
    selected_occupation,
    selected_sex,
    selected_stay_days
):
    return [
        html.H3('入力内容:'),
        html.P('地域: {}'.format(selected_region)),
        html.P('所得: {}'.format(selected_income)),
        html.P('訪問回数: {}'.format(selected_frequency)),
        html.P('交通手段: {}'.format(selected_transport)),
        html.P('職業: {}'.format(selected_occupation)),
        html.P('性別: {}'.format(selected_sex)),
        html.P('滞在日数: {}'.format(selected_stay_days))
    ]

# アプリケーションの実行
if __name__ == '__main__':
    app.run_server(debug=True)
