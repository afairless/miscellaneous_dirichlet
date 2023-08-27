#! usr/bin/env python3

from pathlib import Path

import numpy as np
from scipy.stats import beta
from scipy.stats import dirichlet

import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output


def read_dirichlet_parameters() -> list[float]:
    """
    Read the alpha parameters for the Dirichlet distribution
    The number of parameters equals the number of dimensions of the Dirichlet 
        distribution
    """

    input_filename = 'dirichlet_alpha_parameters.csv'
    input_filepath = list(Path('.').rglob('*/' + input_filename))[0]

    with open(input_filepath) as param_file:
        file_string = param_file.read().replace('\n', '')
        param_file.close()

    alpha = [float(e) for e in file_string.split(',')]

    return alpha 


def read_show_beta_plot_parameters() -> bool:
    """
    If the configuration file contains a single '1', then the marginal beta 
        plots are shown; otherwise, they are hidden

    return:  Boolean indicating whether or not to show the beta plots
    """

    input_filename = 'show_beta_plot_parameter.txt'
    input_filepath = list(Path('.').rglob('*/' + input_filename))[0]

    with open(input_filepath) as param_file:
        file_string = param_file.read().replace('\n', '')
        param_file.close()

    if file_string == '1':
        show_plot = True
    else:
        show_plot = False

    return show_plot 


def beta_statistical_attributes() -> tuple[np.ndarray, float, int]:

    # the beta distribution is defined over the interval [0, 1]
    x = np.arange(0, 1.01, 0.01)

    threshold = 0.50
    idx50 = int(threshold * len(x))

    return x, threshold, idx50 


def beta_plot_attributes() -> tuple[str, str, dict, dict, str, go.Layout]:

    left_color = 'green'
    right_color = 'blue'
    line01 = {'color': left_color, 'width': 5}
    line02 = {'color': right_color, 'width': 5}

    title = (
        'beta beta_id<br>'
        '<sup>'
        f'<span style="color:{left_color}">beta_prop_text0</span>     '
        f'<span style="color:{right_color}">beta_prop_text1</span>'
        '</sup>')

    layout = go.Layout({
        'template': 'plotly_white',
        'showlegend': False,
        'font_size': 24})

    return left_color, right_color, line01, line02, title, layout


app = dash.Dash()

# layout from:  https://community.plotly.com/t/two-graphs-side-by-side/5312/2
app.layout = dash.html.Div([

    dash.html.Div([dash.html.H1(id='heading', style={'textAlign': 'center'})]),

    # add some extra space between title and elements below it
    #dash.html.Div([dash.html.H1(id='placeholder', style={'color': 'white'})]),

    dash.html.Div([
        dash.html.Div(
            id='beta_plot_01_container', 
            children=[dash.dcc.Graph(
                id='beta_plot_01', 
                style={
                    'width': '60vh', 
                    'height': '40vh', 
                    'transform': 'rotate(-45deg)'})], 
            className="four columns"),
        dash.html.Div(
            children=[dash.dcc.Graph(
                id='dirichlet_3d', 
                style={'width': '55vh', 'height': '55vh'})],
            className="four columns"),
        dash.html.Div(
            id='beta_plot_03_container', 
            children=[dash.dcc.Graph(
                id='beta_plot_03', 
                style={
                    'width': '60vh', 
                    'height': '40vh', 
                    'transform': 'rotate(45deg)'})],
            className="four columns"),
    ], className="row"),

    dash.html.Div([
        dash.html.Div(
            id='beta_plot_02_container', 
            children=[dash.dcc.Graph(
                id='beta_plot_02', 
                style={'width': '60vh', 'height': '40vh'})],
            className="offset-by-three four columns"),
    ], className="row"),

    dash.dcc.Interval(id='interval-component', interval=1_000, n_intervals=0)
])

# this CSS file placed inside 'assets' directory within the dash app directory
#app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

# reportedly an updated way to load an external CSS; not clear whether it works
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


@app.callback(
    Output('heading', 'children'),
    Input('interval-component', 'n_intervals'))
def overall_heading(n_intervals: int):
    alpha = [str(e) for e in read_dirichlet_parameters()]
    alpha = ', '.join(alpha)
    return f'Dirichlet alpha: {alpha}'


@app.callback(
    Output('dirichlet_3d', 'figure'),
    Input('interval-component', 'n_intervals'))
def plot_dirichlet_3d(n_intervals: int):

    alpha = read_dirichlet_parameters()
    sample = dirichlet.rvs(alpha, size=100, random_state=61845)

    def makeAxis(title, tickangle):
        return {
          'title': title,
          'titlefont': { 'size': 40 },
          'tickangle': tickangle,
          'tickfont': { 'size': 15 },
          'tickcolor': 'rgba(0,0,0,0)',
          'ticklen': 2,
          'showline': True,
          'showgrid': True
        }

    fig_trace = go.Scatterternary({
        'mode': 'markers',
        'a': sample[:, 0],
        'b': sample[:, 1],
        'c': sample[:, 2],
        'marker': {
            #'symbol': 100,
            'color': '#DB7365',
            'size': 8,
            #'line': { 'width': 2 }
            'opacity': 0.5}
    })

    layout01 = {'template': 'plotly_white'}

    layout02 = {
        'ternary': {
            'sum': 1,
            'aaxis': makeAxis('A', 0),
            'baxis': makeAxis('B', 45),
            'caxis': makeAxis('C', -45)
        },
    }

    fig = go.Figure(data=fig_trace, layout=layout01)
    fig.update_layout(layout02)

    return fig


@app.callback(
    Output('beta_plot_01_container', 'style'),
    Input('interval-component', 'n_intervals'))
def show_plot_beta_01(n_intervals: int):

    show_plot = read_show_beta_plot_parameters()

    if show_plot:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('beta_plot_01', 'figure'),
    Input('interval-component', 'n_intervals'))
def plot_beta_01(n_intervals: int):

    alpha = read_dirichlet_parameters()
    beta_params = [alpha[0], sum(alpha) - alpha[0]]

    x, threshold, idx50 = beta_statistical_attributes()
    left_color, right_color, line01, line02, title, layout = beta_plot_attributes()

    y = beta.pdf(x, beta_params[0], beta_params[1])

    beta_prop0 = beta.cdf(threshold, beta_params[0], beta_params[1])
    beta_prop1 = 1 - beta_prop0 
    assert isinstance(beta_prop0, np.float64)
    assert isinstance(beta_prop1, np.float64)
    beta_prop_text0 = str(round(beta_prop0, 2))
    beta_prop_text1 = str(round(beta_prop1, 2))

    title = title.replace('beta_id', 'A')
    title = title.replace('beta_prop_text0', beta_prop_text0)
    title = title.replace('beta_prop_text1', beta_prop_text1)

    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=x[:idx50+1], y=y[:idx50+1], line=line01, fill='tozeroy'))
    fig.add_trace(go.Scatter(x=x[idx50:], y=y[idx50:], line=line02, fill='tozeroy'))
    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='black', tick0=0, dtick=0.2)
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor='black', showticklabels=False)
    fig.update_layout(title=title, title_x=0.5)

    return fig


@app.callback(
    Output('beta_plot_02_container', 'style'),
    Input('interval-component', 'n_intervals'))
def show_plot_beta_02(n_intervals: int):

    show_plot = read_show_beta_plot_parameters()

    if show_plot:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('beta_plot_02', 'figure'),
    Input('interval-component', 'n_intervals'))
def plot_beta_02(n_intervals: int):

    alpha = read_dirichlet_parameters()
    beta_params = [alpha[1], sum(alpha) - alpha[1]]

    x, threshold, idx50 = beta_statistical_attributes()
    left_color, right_color, line01, line02, title, layout = beta_plot_attributes()

    y = beta.pdf(x, beta_params[0], beta_params[1])

    beta_prop0 = beta.cdf(threshold, beta_params[0], beta_params[1])
    beta_prop1 = 1 - beta_prop0 
    assert isinstance(beta_prop0, np.float64)
    assert isinstance(beta_prop1, np.float64)
    beta_prop_text0 = str(round(beta_prop0, 2))
    beta_prop_text1 = str(round(beta_prop1, 2))

    title = title.replace('beta_id', 'B')
    title = title.replace('beta_prop_text0', beta_prop_text0)
    title = title.replace('beta_prop_text1', beta_prop_text1)
    title = title.replace(left_color, 'dummy')
    title = title.replace(right_color, left_color)
    title = title.replace('dummy', right_color)

    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=x[:idx50+1], y=y[:idx50+1], line=line01, fill='tozeroy'))
    fig.add_trace(go.Scatter(x=x[idx50:], y=y[idx50:], line=line02, fill='tozeroy'))
    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='black', tick0=0, dtick=0.2, autorange='reversed')
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor='black', showticklabels=False)
    fig.update_layout(title=title, title_x=0.5)

    return fig


@app.callback(
    Output('beta_plot_03_container', 'style'),
    Input('interval-component', 'n_intervals'))
def show_plot_beta_03(n_intervals: int):

    show_plot = read_show_beta_plot_parameters()

    if show_plot:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('beta_plot_03', 'figure'),
    Input('interval-component', 'n_intervals'))
def plot_beta_03(n_intervals: int):

    alpha = read_dirichlet_parameters()
    beta_params = [alpha[2], sum(alpha) - alpha[2]]

    x, threshold, idx50 = beta_statistical_attributes()
    left_color, right_color, line01, line02, title, layout = beta_plot_attributes()

    y = beta.pdf(x, beta_params[0], beta_params[1])

    beta_prop0 = beta.cdf(threshold, beta_params[0], beta_params[1])
    beta_prop1 = 1 - beta_prop0 
    assert isinstance(beta_prop0, np.float64)
    assert isinstance(beta_prop1, np.float64)
    beta_prop_text0 = str(round(beta_prop0, 2))
    beta_prop_text1 = str(round(beta_prop1, 2))

    title = title.replace('beta_id', 'C')
    title = title.replace('beta_prop_text0', beta_prop_text0)
    title = title.replace('beta_prop_text1', beta_prop_text1)

    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=x[:idx50+1], y=y[:idx50+1], line=line01, fill='tozeroy'))
    fig.add_trace(go.Scatter(x=x[idx50:], y=y[idx50:], line=line02, fill='tozeroy'))
    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='black', tick0=0, dtick=0.2)#autorange='reversed')
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor='black', showticklabels=False)
    fig.update_layout(title=title, title_x=0.5)

    return fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
