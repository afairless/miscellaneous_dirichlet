#! usr/bin/env python3

from typing import Sequence

import numpy as np
from scipy.stats import beta, dirichlet, multinomial

import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output, State
from dash_daq import BooleanSwitch


def calculate_dirichlet_mode(alpha: Sequence[float]) -> np.ndarray:
    """
    """

    alpha_sum = sum(alpha)
    alpha_len = len(alpha)
    dirichlet_mode = [(e - 1) / (alpha_sum - alpha_len) for e in alpha]

    return np.array(dirichlet_mode)


def generate_multinomial_data(
    true_alpha_mode: np.ndarray, data_points_n: int=100, 
    random_state: int=31723, return_one_hot_encoded: bool=True) -> np.ndarray:
    """
    Generates a sample of multinomial data
    """

    assert np.isclose(sum(true_alpha_mode), 1)

    sample = multinomial.rvs(
        1, true_alpha_mode, size=data_points_n, random_state=random_state)

    if return_one_hot_encoded:
        return sample
    else:
        _, idx = np.where(sample)
        return idx


def calculate_dirichlet_parameter_series(
    dirichlet_parameter_alpha: Sequence[float]=[2, 2, 2],
    multinomial_sample: np.ndarray=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    ) -> np.ndarray:
    """
    Given the prior Dirichlet distribution parameters alpha and a multinomially
        distributed sample, calculate the series of Dirichlet parameters for 
        each update of the Dirichlet distribution based on each successive item 
        in the multinomial sample
    """

    param_sums = np.apply_along_axis(np.cumsum, 0, multinomial_sample)
    dirichlet_param_series = dirichlet_parameter_alpha + param_sums 

    return dirichlet_param_series 


def create_barycentric_grid_coordinates(axis_ticks_n: int=100) -> np.ndarray:
    """
    Given the number of coordinates/ticks per axis (range from 0 to 1), returns 
        coordinates of every tick combination for a triangle (barycentric plot)
        in an N x 3 matrix, where the 3 columns correspond to the 3 axes
    """

    coordinates = np.linspace(0, 1, axis_ticks_n)
    coordinate_matrix = np.meshgrid(coordinates, coordinates)
    grid = np.stack(coordinate_matrix).reshape(2, -1).transpose()
    grid = np.hstack((grid, 1 - grid.sum(axis=1).reshape(-1, 1)))

    # remove all grid points not on the simplex, i.e., that don't add up to 1
    coordinate_sums = np.absolute(grid).sum(axis=1)
    grid = grid[coordinate_sums == 1, :]

    return grid


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


# vertical white space between rows of input widgets
padding_space = "15px"

app = dash.Dash()

# layout from:  https://community.plotly.com/t/two-graphs-side-by-side/5312/2
app.layout = dash.html.Div([

    dash.html.Div([
        dash.html.Div(
            children=BooleanSwitch(
                id='show-betas', 
                on=True, 
                label='Show Beta Plots'), 
            className="two columns"),
        dash.html.Div(
            children=dash.html.Button(
                children='Restart Animation', 
                id='interval_reset', 
                n_clicks=0), 
            className="two columns"),
        dash.html.Div(
            children=BooleanSwitch(
                id='pause-toggle', 
                on=True, 
                label='Play Animation'), 
            className="two columns"),
        dash.html.Div(
            children=dash.html.P('Animation Update Frequency (sec)'), 
            className="two columns"),
        dash.html.Div(
            children=dash.dcc.Slider(
                id='update-frequency', min=0.25, max=10, step=0.25, value=1, 
                marks={i: str(i) for i in range(1, 11)},
                tooltip={'placement': 'bottom', 'always_visible': True}, 
                updatemode='drag'),
            className="six columns"),
    ], className="row"),


    dash.html.Div(style={"padding": padding_space}),

    dash.html.Div([
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Dirichlet parameter alpha 1 starting value (prior)')), 
            className="four columns"),
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Dirichlet parameter alpha 2 starting value (prior)')), 
            className="four columns"),
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Dirichlet parameter alpha 2 starting value (prior)')), 
            className="four columns"),
        ], className="row"),

    dash.html.Div([
        dash.html.Div(
            children=dash.dcc.Input(
                id='dirichlet_parameter_1', type='number', 
                min=1.01, max=500, step=0.01, value=2, placeholder='Alpha 1'), 
            className="four columns"),
        dash.html.Div(
            children=dash.dcc.Input(
                id='dirichlet_parameter_2', type='number', 
                min=1.01, max=500, step=0.01, value=2, placeholder='Alpha 2'), 
            className="four columns"),
        dash.html.Div(
            children=dash.dcc.Input(
                id='dirichlet_parameter_3', type='number', 
                min=1.01, max=500, step=0.01, value=2, placeholder='Alpha 3'), 
            className="four columns"),
        ], className="row"),

    dash.html.Div(style={"padding": padding_space}),

    dash.html.Div([
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Number of multinomially distributed data points')), 
            className="two columns"),
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Random seed')), 
            className="two columns"),
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Multinomial Proportion (True Distribution Mode)')), 
            className="two columns"),
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Multinomial Proportion (True Distribution Mode)')), 
            className="two columns"),
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Multinomial Proportion (True Distribution Mode)')), 
            className="two columns"),
        ], className="row"),

    dash.html.Div([
        dash.html.Div(
            children=dash.dcc.Input(
                id='data_points_n', type='number', 
                min=3, max=1000, step=1, value=100, 
                placeholder='Number of Data Points'), 
            className="two columns"),
        dash.html.Div(
            children=dash.dcc.Input(
                id='random_state', type='number', 
                min=101, max=1e8, step=1, value=1e4, placeholder='Random Seed'), 
            className="two columns"),
        dash.html.Div(
            children=dash.dcc.Slider(
                id='binomial_proportion_1', 
                min=0, max=1, step=0.01, value=0.5, marks=None, 
                tooltip={'placement': 'bottom', 'always_visible': True}, 
                updatemode='drag'), 
            className="two columns"),
        dash.html.Div(
            children=dash.dcc.Slider(
                id='binomial_proportion_2', 
                min=0, max=1, step=0.01, value=0.5, marks=None, 
                tooltip={'placement': 'bottom', 'always_visible': True}, 
                updatemode='drag'), 
            className="two columns"),
        dash.html.Div(
            children=dash.dcc.Slider(
                id='binomial_proportion_3', 
                min=0, max=1, step=0.01, value=0.5, marks=None, 
                tooltip={'placement': 'bottom', 'always_visible': True}, 
                updatemode='drag'), 
            className="two columns"),
        ], className="row"),

    dash.html.Div(style={"padding": padding_space}),

    dash.html.Div([dash.html.H1(id='heading', style={'textAlign': 'center'})]),

    dash.html.Div([
        dash.html.Div(
            id='beta_plot_01_container', 
            children=[dash.dcc.Graph(
                id='beta_plot_01', 
                style={
                    'width': '40vh', 
                    'height': '40vh', 
                    'transform': 'rotate(-90deg)'})], 
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
                    'width': '40vh', 
                    'height': '40vh', 
                    'transform': 'rotate(30deg)'})],
            className="four columns"),
    ], className="row"),

    dash.html.Div([
        dash.html.Div(
            id='beta_plot_02_container', 
            children=[dash.dcc.Graph(
                id='beta_plot_02', 
                style={
                    'width': '40vh', 
                    'height': '40vh', 
                    'transform': 'rotate(-30deg)'})],
            className="offset-by-three four columns"),
    ], className="row"),

    dash.dcc.Interval(id='interval-component', interval=1_000, n_intervals=0, disabled=False)
])

# this CSS file placed inside 'assets' directory within the dash app directory
#app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

# reportedly an updated way to load an external CSS; not clear whether it works
#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


@app.callback(
    Output('interval-component', 'disabled'),
    Input('pause-toggle', 'on'))
def pause_animation(enabled: bool) -> bool:
    """
    Boolean user choice of whether to play or pause the animation of updating 
        beta distributions
    """
    return not enabled


@app.callback(
    Output('interval-component', 'interval'),
    Input('update-frequency', 'value'))
def animation_frequency(value: float) -> float:
    """
    User choice of interval specifying how often the animation should be updated
    User chooses value in seconds, which is converted to milliseconds
    """
    return value * 1000


@app.callback(
    Output('interval-component', 'n_intervals'),
    [Input('interval_reset', 'n_clicks'),
     Input('interval-component', 'n_intervals')],
    State('interval_reset', 'value'))
def reset_interval(n_clicks, n_intervals, value) -> int:
    """
    User choice to restart animation from the start of the beta update sequence
    """
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'interval_reset' in changed_id:
        return 0
    else:
        return n_intervals


@app.callback(
    Output('heading', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('dirichlet_parameter_1', 'value'), 
     Input('dirichlet_parameter_2', 'value'), 
     Input('dirichlet_parameter_3', 'value'),
     Input('data_points_n', 'value'),
     Input('random_state', 'value')])
def overall_heading(
    n_intervals: int, dirichlet_parameter_1: float, 
    dirichlet_parameter_2: float, dirichlet_parameter_3: float, 
    data_points_n: int, random_state: int) -> str:
    """
    Displays Dirichlet distribution parameters 
    """

    alpha = [
        dirichlet_parameter_1, dirichlet_parameter_2, dirichlet_parameter_3]
    dirichlet_alpha_mode = calculate_dirichlet_mode(alpha)
    sample = generate_multinomial_data(
        dirichlet_alpha_mode, data_points_n, random_state)
    parameter_series = calculate_dirichlet_parameter_series(alpha, sample)

    loop_len = min(n_intervals, len(parameter_series)-1)
    parameters = parameter_series[loop_len, :] 
    title = f'{n_intervals}, {loop_len}, {len(parameter_series)}, {parameters[0]}, {parameters[1]}, {parameters[2]}'
    #title = f'{parameters[0]}, {parameters[1]}, {parameters[2]}'
    return title


@app.callback(
    Output('dirichlet_3d', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('dirichlet_parameter_1', 'value'), 
     Input('dirichlet_parameter_2', 'value'), 
     Input('dirichlet_parameter_3', 'value'),
     Input('data_points_n', 'value'),
     Input('random_state', 'value')])
def plot_dirichlet_3d(
    n_intervals: int, dirichlet_parameter_1: float,
    dirichlet_parameter_2: float, dirichlet_parameter_3: float,
    data_points_n: int, random_state: int) -> go.Figure:

    alpha = [
        dirichlet_parameter_1, dirichlet_parameter_2, dirichlet_parameter_3]
    dirichlet_alpha_mode = calculate_dirichlet_mode(alpha)
    sample = generate_multinomial_data(
        dirichlet_alpha_mode, data_points_n, random_state)
    parameter_series = calculate_dirichlet_parameter_series(alpha, sample)
    loop_len = min(n_intervals, len(parameter_series)-1)
    parameters = parameter_series[loop_len, :] 

    grid = create_barycentric_grid_coordinates(91)

    # apply probability density function across triangular plot grid
    pdfs = [dirichlet.pdf(e, parameters) for e in grid]
    
    dirichlet_mode = calculate_dirichlet_mode(parameters)

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
        'a': grid[:, 0],
        'b': grid[:, 1],
        'c': grid[:, 2],
        'marker': {
            #'symbol': 100,
            'color': '#DB7365',
            #'color': ['red', 'blue', 'yellow', 'green'],
            'color': pdfs,
            'size': 8,
            #'line': { 'width': 2 }
            'opacity': 0.9}
    })

    fig_trace_mode = go.Scatterternary({
        'mode': 'markers',
        'a': np.array(dirichlet_mode[0]),
        'b': np.array(dirichlet_mode[1]),
        'c': np.array(dirichlet_mode[2]),
        'marker': {
            #'symbol': 100,
            'color': 'blue',
            'size': 8,
            #'line': { 'width': 2 }
            'opacity': 0.5}
    })

    layout01 = {
        'template': 'plotly_white',
        'showlegend': False}

    layout02 = {
        'ternary': {
            'sum': 1,
            'aaxis': makeAxis('A', 0),
            'baxis': makeAxis('B', 45),
            'caxis': makeAxis('C', -45)
        },
    }

    fig = go.Figure(data=fig_trace, layout=layout01)
    fig.add_trace(fig_trace_mode)
    fig.update_layout(layout02)

    return fig


@app.callback(
    Output('beta_plot_01_container', 'style'),
    Input('show-betas', 'on'))
def show_plot_beta_01(show_beta_plots: bool):

    if show_beta_plots:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('beta_plot_01', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('dirichlet_parameter_1', 'value'), 
     Input('dirichlet_parameter_2', 'value'), 
     Input('dirichlet_parameter_3', 'value')])
def plot_beta_01(n_intervals: int,
        dirichlet_parameter_1: float,
        dirichlet_parameter_2: float,
        dirichlet_parameter_3: float,
        ):

    alpha = [dirichlet_parameter_1, dirichlet_parameter_2, dirichlet_parameter_3]
    beta_params = [alpha[0], sum(alpha) - alpha[0]]
    beta_mode = (beta_params[0] - 1) / (beta_params[0] + beta_params[1] - 2)

    x, threshold, idx50 = beta_statistical_attributes()
    left_color, right_color, line01, line02, title, layout = beta_plot_attributes()

    y = beta.pdf(x, beta_params[0], beta_params[1])

    beta_prop0 = beta.cdf(threshold, beta_params[0], beta_params[1])
    beta_prop1 = 1 - beta_prop0 
    beta_prop_text0 = str(round(beta_prop0, 2))
    beta_prop_text1 = str(round(beta_prop1, 2))
    beta_mode_text = str(round(beta_mode, 3))

    title = title.replace('beta_id', 'A')
    title = title.replace('beta_prop_text0', beta_mode_text)
    #title = title.replace('beta_prop_text0', beta_prop_text0)
    #title = title.replace('beta_prop_text1', beta_prop_text1)
    title = title.replace('beta_prop_text1', '')

    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=x[:idx50+1], y=y[:idx50+1], line=line01, fill='tozeroy'))
    fig.add_trace(go.Scatter(x=x[idx50:], y=y[idx50:], line=line02, fill='tozeroy'))
    fig.add_vline(x=beta_mode)
    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='black', tick0=0, dtick=0.2)
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor='black', showticklabels=False)
    fig.update_layout(title=title, title_x=0.5)

    return fig


@app.callback(
    Output('beta_plot_02_container', 'style'),
    Input('show-betas', 'on'))
def show_plot_beta_02(show_beta_plots: bool):

    if show_beta_plots:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('beta_plot_02', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('dirichlet_parameter_1', 'value'), 
     Input('dirichlet_parameter_2', 'value'), 
     Input('dirichlet_parameter_3', 'value')])
def plot_beta_02(n_intervals: int,
        dirichlet_parameter_1: float,
        dirichlet_parameter_2: float,
        dirichlet_parameter_3: float,
        ):

    alpha = [dirichlet_parameter_1, dirichlet_parameter_2, dirichlet_parameter_3]
    beta_params = [alpha[1], sum(alpha) - alpha[1]]
    beta_mode = (beta_params[0] - 1) / (beta_params[0] + beta_params[1] - 2)

    x, threshold, idx50 = beta_statistical_attributes()
    left_color, right_color, line01, line02, title, layout = beta_plot_attributes()

    y = beta.pdf(x, beta_params[0], beta_params[1])

    beta_prop0 = beta.cdf(threshold, beta_params[0], beta_params[1])
    beta_prop1 = 1 - beta_prop0 
    beta_prop_text0 = str(round(beta_prop0, 2))
    beta_prop_text1 = str(round(beta_prop1, 2))
    beta_mode_text = str(round(beta_mode, 3))

    title = title.replace('beta_id', 'B')
    title = title.replace('beta_prop_text0', beta_mode_text)
    title = title.replace('beta_prop_text1', '')
    #title = title.replace('beta_prop_text0', beta_prop_text0)
    #title = title.replace('beta_prop_text1', beta_prop_text1)
    #title = title.replace(left_color, 'dummy')
    #title = title.replace(right_color, left_color)
    #title = title.replace('dummy', right_color)

    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=x[:idx50+1], y=y[:idx50+1], line=line01, fill='tozeroy'))
    fig.add_trace(go.Scatter(x=x[idx50:], y=y[idx50:], line=line02, fill='tozeroy'))
    fig.add_vline(x=beta_mode)
    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='black', tick0=0, dtick=0.2, autorange='reversed')
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor='black', showticklabels=False)
    fig.update_layout(title=title, title_x=0.5)

    return fig


@app.callback(
    Output('beta_plot_03_container', 'style'),
    Input('show-betas', 'on'))
def show_plot_beta_03(show_beta_plots: bool):

    if show_beta_plots:
        return {'display': 'block'}
    else:
        return {'display': 'none'}


@app.callback(
    Output('beta_plot_03', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('dirichlet_parameter_1', 'value'), 
     Input('dirichlet_parameter_2', 'value'), 
     Input('dirichlet_parameter_3', 'value')])
def plot_beta_03(n_intervals: int,
        dirichlet_parameter_1: float,
        dirichlet_parameter_2: float,
        dirichlet_parameter_3: float,
        ):

    alpha = [dirichlet_parameter_1, dirichlet_parameter_2, dirichlet_parameter_3]
    beta_params = [alpha[2], sum(alpha) - alpha[2]]
    beta_mode = (beta_params[0] - 1) / (beta_params[0] + beta_params[1] - 2)

    x, threshold, idx50 = beta_statistical_attributes()
    left_color, right_color, line01, line02, title, layout = beta_plot_attributes()

    y = beta.pdf(x, beta_params[0], beta_params[1])

    beta_prop0 = beta.cdf(threshold, beta_params[0], beta_params[1])
    beta_prop1 = 1 - beta_prop0 
    beta_prop_text0 = str(round(beta_prop0, 2))
    beta_prop_text1 = str(round(beta_prop1, 2))
    beta_mode_text = str(round(beta_mode, 3))

    title = title.replace('beta_id', 'C')
    title = title.replace('beta_prop_text0', beta_mode_text)
    #title = title.replace('beta_prop_text0', beta_prop_text0)
    #title = title.replace('beta_prop_text1', beta_prop_text1)
    title = title.replace('beta_prop_text1', '')

    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=x[:idx50+1], y=y[:idx50+1], line=line01, fill='tozeroy'))
    fig.add_trace(go.Scatter(x=x[idx50:], y=y[idx50:], line=line02, fill='tozeroy'))
    fig.add_vline(x=beta_mode)
    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='black', tick0=0, dtick=0.2)#autorange='reversed')
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor='black', showticklabels=False)
    fig.update_layout(title=title, title_x=0.5)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
