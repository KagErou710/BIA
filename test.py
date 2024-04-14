import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Application definition
app = dash.Dash(__name__)

# Layout definition
app.layout = html.Div(children=[
    html.H1('Multiple Textbox Input and Display'),
    html.Div(children=[
        html.Label('Textbox 1:'),
        dcc.Input(id='input-1', type='text', value=''),
    ]),
    html.Div(children=[
        html.Label('Textbox 2:'),
        dcc.Input(id='input-2', type='text', value=''),
    ]),
    html.Div(children=[
        html.Label('Textbox 3:'),
        dcc.Input(id='input-3', type='text', value=''),
    ]),
    html.Button('Display', id='button'),
    html.Div(id='output-container', children=[])  # Input display area
])

# Callback function definition
@app.callback(
    Output('output-container', 'children'),
    [Input('button', 'n_clicks'),
     Input('input-1', 'value'),
     Input('input-2', 'value'),
     Input('input-3', 'value')]
)
def update_output_container(n_clicks, input1, input2, input3):
    if n_clicks is None:
        return []  # Initially, display nothing

    # Gather input texts
    input_texts = [input1, input2, input3]
    input_texts = [text for text in input_texts if text]  # Exclude empty strings

    # Combine and display input texts
    if input_texts:
        output_text = f'Entered Texts: {", ".join(input_texts)}'
    else:
        output_text = 'No inputs yet'

    return [html.P(output_text)]

# Application execution
if __name__ == '__main__':
    app.run_server(debug=True)
