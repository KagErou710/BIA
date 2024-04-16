from dash import Dash, html, dcc, dash_html_components as html_comp

app = Dash(__name__)

app.layout = html_comp.Div([
    html_comp.Div([
        html_comp.Div([
            html_comp.Input(id="input1", type="text", placeholder="入力欄1"),
            html_comp.Input(id="input2", type="text", placeholder="入力欄2"),
        ], style={'display': 'flex', 'flex-direction': 'column'}),
        html_comp.Div([
            html_comp.Img(id="image", src="https://placehold.co/600x400", height=400),
        ], style={'width': '60%', 'height': '400px'}),
        html_comp.Div([
            html_comp.Button("ボタン", id="button"),
        ], style={'width': '40%', 'height': '400px'}),
    ], style={'display': 'flex', 'flex-direction': 'row', 'height': '400px'}),
])

if __name__ == '__main__':
    app.run_server(debug=True)
