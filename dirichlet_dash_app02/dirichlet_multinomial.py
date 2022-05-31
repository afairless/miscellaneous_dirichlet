#! usr/bin/env python3

from typing import Sequence

import numpy as np
from scipy.stats import beta, dirichlet, multinomial

import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output, State
from dash_daq import BooleanSwitch


def calculate_dirichlet_mode_from_user_input(
    multinomial_proportion_1: float, multinomial_proportion_2: float
    ) -> tuple[float, float, float]:
    """
    Calculates true Dirichlet mode based on user input for 2 of the 3 values

    The Dirichlet mode must be on the simplex, i.e., all values must sum to 1

    If user input is incompatible with the simplex, then a default is returned
    """

    input_multinomial_sum = multinomial_proportion_1 + multinomial_proportion_2

    if input_multinomial_sum > 1:
        return (1/3, 1/3, 1/3)

    else: 
        multinomial_proportion_3 = 1 - input_multinomial_sum 
        return (
            multinomial_proportion_1, 
            multinomial_proportion_2,
            multinomial_proportion_3)


def generate_multinomial_data(
    true_alpha_mode: Sequence[float], data_points_n: int=100, 
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


def calculate_dirichlet_mode(alpha: Sequence[float]) -> np.ndarray:
    """
    """

    alpha_sum = sum(alpha)
    alpha_len = len(alpha)
    dirichlet_mode = [(e - 1) / (alpha_sum - alpha_len) for e in alpha]

    return np.array(dirichlet_mode)


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


def calculate_beta_parameter_series(
    beta_parameter_alpha: float, 
    beta_parameter_beta: float, 
    binomial_sample: np.ndarray) -> list[tuple[float, float]]:
    """
    Given the prior beta distribution parameters alpha and beta and a binomially
        distributed sample, calculate the series of beta parameters for each
        update of the beta distribution based on each successive item in the 
        binomial sample
    """

    # if user provides no input values, assign default value
    if not beta_parameter_alpha:
        beta_parameter_alpha = 2
    if not beta_parameter_beta:
        beta_parameter_beta = 2

    alpha_param_sums = binomial_sample.cumsum() 
    beta_param_sums = range(len(alpha_param_sums)) - alpha_param_sums + 1
    beta_param_series = [
        (beta_parameter_alpha + alpha_param_sums[i], 
         beta_parameter_beta + beta_param_sums[i]) 
        for i in range(len(binomial_sample))]
    beta_param_series = (
        [(beta_parameter_alpha, beta_parameter_beta)] + beta_param_series)

    return beta_param_series


def calculate_beta_mode(
    beta_parameter_alpha: float, beta_parameter_beta: float) -> float:

    beta_mode = (
        (beta_parameter_alpha  - 1) / 
        (beta_parameter_alpha + beta_parameter_beta - 2))

    return beta_mode


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
                dash.html.P('Dirichlet parameter starting values (priors):')), 
            className="two columns"),
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Alpha A')), 
            className="one columns"),
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Alpha B')), 
            className="one columns"),
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Alpha C')), 
            className="one columns"),
        ], className="row"),

    dash.html.Div([
        dash.html.Div(
            children=dash.dcc.Input(
                id='dirichlet_parameter_1', type='number', 
                min=1.01, max=500, step=0.01, value=2, placeholder='Alpha 1'), 
            className="offset-by-two one columns"),
        dash.html.Div(
            children=dash.dcc.Input(
                id='dirichlet_parameter_2', type='number', 
                min=1.01, max=500, step=0.01, value=2, placeholder='Alpha 2'), 
            className="one columns"),
        dash.html.Div(
            children=dash.dcc.Input(
                id='dirichlet_parameter_3', type='number', 
                min=1.01, max=500, step=0.01, value=2, placeholder='Alpha 3'), 
            className="one columns"),
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
                dash.html.P('Multinomial Proportion (True Distribution Mode):')), 
            className="two columns"),
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Proportion A')), 
            className="two columns"),
        dash.html.Div(
            children=dash.html.Div(
                dash.html.P('Proportion B')), 
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
                id='multinomial_proportion_1', 
                min=0, max=1, step=0.01, value=1/3, marks=None, 
                tooltip={'placement': 'bottom', 'always_visible': True}, 
                updatemode='drag'), 
            className="offset-by-two two columns"),
        dash.html.Div(
            children=dash.dcc.Slider(
                id='multinomial_proportion_2', 
                min=0, max=1, step=0.01, value=1/3, marks=None, 
                tooltip={'placement': 'bottom', 'always_visible': True}, 
                updatemode='drag'), 
            className="two columns"),
        ], className="row"),

    dash.html.Div(style={"padding": padding_space}),

    dash.html.Div([dash.html.H2(id='heading1', style={'textAlign': 'center'})]),
    dash.html.Div([dash.html.H2(id='heading2', style={'textAlign': 'center'})]),
    dash.html.Div([dash.html.H2(id='heading3', style={'textAlign': 'center'})]),

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
            className="offset-by-four four columns"),
    ], className="row"),

    dash.dcc.Interval(
        id='interval-component', interval=1_000, n_intervals=0, disabled=False)
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
    Output('heading1', 'children'),
    [Input('multinomial_proportion_1', 'value'),
     Input('multinomial_proportion_2', 'value')])
def overall_heading1(
    multinomial_proportion_1: float, multinomial_proportion_2: float) -> str: 
    """
    Displays Dirichlet distribution parameters 
    """

    # generate multinomial sample based on user's specified mode/proportions
    true_alpha_mode = calculate_dirichlet_mode_from_user_input(
        multinomial_proportion_1, multinomial_proportion_2)

    title = (
        f'Mode: A ={true_alpha_mode[0]: .2f}, '
        f'B ={true_alpha_mode[1]: .2f}, '
        f'C ={true_alpha_mode[2]: .2f}')

    return title


@app.callback(
    Output('heading2', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('dirichlet_parameter_1', 'value'), 
     Input('dirichlet_parameter_2', 'value'), 
     Input('dirichlet_parameter_3', 'value'),
     Input('multinomial_proportion_1', 'value'),
     Input('multinomial_proportion_2', 'value'),
     Input('data_points_n', 'value'),
     Input('random_state', 'value')])
def overall_heading2(
    n_intervals: int, dirichlet_parameter_1: float, 
    dirichlet_parameter_2: float, dirichlet_parameter_3: float, 
    multinomial_proportion_1: float, multinomial_proportion_2: float, 
    data_points_n: int, random_state: int) -> str:
    """
    Displays Dirichlet distribution parameters 
    """

    # generate multinomial sample based on user's specified mode/proportions
    true_alpha_mode = calculate_dirichlet_mode_from_user_input(
        multinomial_proportion_1, multinomial_proportion_2)
    sample = generate_multinomial_data(
        true_alpha_mode, data_points_n, random_state)

    # calculate Dirichlet parameters after each update from the sample 
    alpha = (
        dirichlet_parameter_1, dirichlet_parameter_2, dirichlet_parameter_3)
    dirichlet_parameter_series = calculate_dirichlet_parameter_series(
        alpha, sample)

    # calculate Dirichlet statistics for the current update
    loop_len = min(n_intervals, len(dirichlet_parameter_series)-1)
    dirichlet_parameters = dirichlet_parameter_series[loop_len, :] 
    dirichlet_mode = calculate_dirichlet_mode(dirichlet_parameters)

    title = (
        #f'{n_intervals}, {loop_len}, {len(dirichlet_parameter_series)}; '
        #f'{loop_len}; '
        f'Estimate: A ={dirichlet_mode[0]: .2f}, '
        f'B={dirichlet_mode[1]: .2f}, '
        f'C ={dirichlet_mode[2]: .2f}')

    return title


@app.callback(
    Output('heading3', 'children'),
    [Input('interval-component', 'n_intervals'),
     Input('dirichlet_parameter_1', 'value'), 
     Input('dirichlet_parameter_2', 'value'), 
     Input('dirichlet_parameter_3', 'value'),
     Input('multinomial_proportion_1', 'value'),
     Input('multinomial_proportion_2', 'value'),
     Input('data_points_n', 'value'),
     Input('random_state', 'value')])
def overall_heading3(
    n_intervals: int, dirichlet_parameter_1: float, 
    dirichlet_parameter_2: float, dirichlet_parameter_3: float, 
    multinomial_proportion_1: float, multinomial_proportion_2: float, 
    data_points_n: int, random_state: int) -> str:
    """
    Displays Dirichlet distribution parameters 
    """

    # generate multinomial sample based on user's specified mode/proportions
    true_alpha_mode = calculate_dirichlet_mode_from_user_input(
        multinomial_proportion_1, multinomial_proportion_2)
    sample = generate_multinomial_data(
        true_alpha_mode, data_points_n, random_state)

    # calculate Dirichlet parameters after each update from the sample 
    alpha = (
        dirichlet_parameter_1, dirichlet_parameter_2, dirichlet_parameter_3)
    dirichlet_parameter_series = calculate_dirichlet_parameter_series(
        alpha, sample)

    # calculate Dirichlet statistics for the current update
    loop_len = min(n_intervals, len(dirichlet_parameter_series)-1)
    dirichlet_parameters = dirichlet_parameter_series[loop_len, :] 

    title = (
        f'Parameters: A = {dirichlet_parameters[0]}, '
        f'B = {dirichlet_parameters[1]}, '
        f'C = {dirichlet_parameters[2]}')

    return title


@app.callback(
    Output('dirichlet_3d', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('dirichlet_parameter_1', 'value'), 
     Input('dirichlet_parameter_2', 'value'), 
     Input('dirichlet_parameter_3', 'value'),
     Input('multinomial_proportion_1', 'value'),
     Input('multinomial_proportion_2', 'value'),
     Input('data_points_n', 'value'),
     Input('random_state', 'value')])
def plot_dirichlet_3d(
    n_intervals: int, dirichlet_parameter_1: float,
    dirichlet_parameter_2: float, dirichlet_parameter_3: float,
    multinomial_proportion_1: float, multinomial_proportion_2: float, 
    data_points_n: int, random_state: int) -> go.Figure:

    # generate multinomial sample based on user's specified mode/proportions
    true_alpha_mode = calculate_dirichlet_mode_from_user_input(
        multinomial_proportion_1, multinomial_proportion_2)
    sample = generate_multinomial_data(
        true_alpha_mode, data_points_n, random_state)

    # calculate Dirichlet parameters after each update from the sample 
    alpha = (
        dirichlet_parameter_1, dirichlet_parameter_2, dirichlet_parameter_3)
    dirichlet_parameter_series = calculate_dirichlet_parameter_series(
        alpha, sample)

    # calculate Dirichlet statistics for the current update
    loop_len = min(n_intervals, len(dirichlet_parameter_series)-1)
    dirichlet_parameters = dirichlet_parameter_series[loop_len, :] 
    dirichlet_mode = calculate_dirichlet_mode(dirichlet_parameters)

    grid = create_barycentric_grid_coordinates(91)

    # apply probability density function across triangular plot grid
    pdfs = [dirichlet.pdf(e, dirichlet_parameters) for e in grid]
    
    def makeAxis(title, tickangle, titlefont):
        return {
          'title': title,
          'titlefont': titlefont,
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

    fig_trace_true_mode = go.Scatterternary({
        'mode': 'markers',
        'a': np.array(true_alpha_mode[0]),
        'b': np.array(true_alpha_mode[1]),
        'c': np.array(true_alpha_mode[2]),
        'marker': {
            #'symbol': 100,
            'color': 'black',
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

    # enlarge label of category of the most recent sample update
    category_title_sizes = sample[loop_len, :].copy()
    category_title_sizes[category_title_sizes==0] = 40
    category_title_sizes[category_title_sizes==1] = 70

    layout02 = {
        'ternary': {
            'sum': 1,
            'aaxis': makeAxis('A', 0, {'size': category_title_sizes[0]}),
            'baxis': makeAxis('B', 45, {'size': category_title_sizes[1]}),
            'caxis': makeAxis('C', -45, {'size': category_title_sizes[2]})
        },
    }

    fig = go.Figure(data=fig_trace, layout=layout01)
    fig.add_trace(fig_trace_true_mode)
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
     Input('dirichlet_parameter_3', 'value'),
     Input('multinomial_proportion_1', 'value'),
     Input('multinomial_proportion_2', 'value'),
     Input('data_points_n', 'value'),
     Input('random_state', 'value')])
def plot_beta_01(
    n_intervals: int, dirichlet_parameter_1: float,
    dirichlet_parameter_2: float, dirichlet_parameter_3: float,
    multinomial_proportion_1: float, multinomial_proportion_2: float, 
    data_points_n: int, random_state: int) -> go.Figure:
    """
    Plots sequence of beta distribution updates with more recent distributions
        shown with thicker and more opaque curves
    Specifies the true, user-specified proportion of the binomial distribution
        as well as the most recent beta distribution's beta estimate (i.e., the
        mode of the distribution) of that proportion
    """

    # generate multinomial sample based on user's specified mode/proportions
    true_alpha_mode = calculate_dirichlet_mode_from_user_input(
        multinomial_proportion_1, multinomial_proportion_2)
    sample = generate_multinomial_data(
        true_alpha_mode, data_points_n, random_state)

    # calculate Dirichlet parameters after each update from the sample 
    alpha = (
        dirichlet_parameter_1, dirichlet_parameter_2, dirichlet_parameter_3)
    dirichlet_parameter_series = calculate_dirichlet_parameter_series(
        alpha, sample)

    # calculate beta parameters after each update from the sample 
    beta_idx = 0
    beta_parameter_series = np.column_stack(
        (dirichlet_parameter_series[:, beta_idx],
         dirichlet_parameter_series.sum(axis=1) - 
         dirichlet_parameter_series[:, beta_idx]))
    loop_len = min(n_intervals, len(beta_parameter_series)-1)
    beta_parameters = beta_parameter_series[loop_len, :] 

    true_beta_mode = true_alpha_mode[beta_idx] 
    x, _, _ = beta_statistical_attributes()

    layout = go.Layout({
        'template': 'plotly_white',
        'showlegend': False,
        'font_size': 24})
    max_line_width = 5
    trace_color = 'red'

    fig = go.Figure(layout=layout)

    fig.add_vline(x=true_beta_mode)

    loop_len = min(n_intervals+1, len(beta_parameter_series))
    for i in range(loop_len):

        # make the most recently added lines/traces thicker than older traces
        line_width = [
            max(1, max_line_width - (loop_len - j - 1)) 
            for j in range(loop_len)]
        line01 = {'color': trace_color, 'width': line_width[i]}

        # make the most recently added lines/traces more opaque than older traces
        line_opacity = np.linspace(0.3, 1, loop_len)
        if len(line_opacity) == 1:
            line_opacity = [1]

        # calculate density for each beta distribution
        beta_parameter_alpha = beta_parameter_series[i][0]
        beta_parameter_beta = beta_parameter_series[i][1]
        y = beta.pdf(x, beta_parameter_alpha, beta_parameter_beta)

        fig.add_trace(go.Scatter(x=x, y=y, line=line01, opacity=line_opacity[i]))

    beta_mode = calculate_beta_mode(beta_parameters[0], beta_parameters[1])

    fig.add_vline(x=beta_mode, line_color=trace_color)
    beta_mode = round(beta_mode, 2)
    title = (f'A Mode ={true_beta_mode: .2f}<br><span style="color:red">'
             f'Estimate ={beta_mode: .2f}</span>')

    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='black', tick0=0, dtick=0.2, 
        tickangle=90)
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
     Input('dirichlet_parameter_3', 'value'),
     Input('multinomial_proportion_1', 'value'),
     Input('multinomial_proportion_2', 'value'),
     Input('data_points_n', 'value'),
     Input('random_state', 'value')])
def plot_beta_02(
    n_intervals: int, dirichlet_parameter_1: float,
    dirichlet_parameter_2: float, dirichlet_parameter_3: float,
    multinomial_proportion_1: float, multinomial_proportion_2: float, 
    data_points_n: int, random_state: int) -> go.Figure:
    """
    Plots sequence of beta distribution updates with more recent distributions
        shown with thicker and more opaque curves
    Specifies the true, user-specified proportion of the binomial distribution
        as well as the most recent beta distribution's beta estimate (i.e., the
        mode of the distribution) of that proportion
    """


    # generate multinomial sample based on user's specified mode/proportions
    true_alpha_mode = calculate_dirichlet_mode_from_user_input(
        multinomial_proportion_1, multinomial_proportion_2)
    sample = generate_multinomial_data(
        true_alpha_mode, data_points_n, random_state)

    # calculate Dirichlet parameters after each update from the sample 
    alpha = (
        dirichlet_parameter_1, dirichlet_parameter_2, dirichlet_parameter_3)
    dirichlet_parameter_series = calculate_dirichlet_parameter_series(
        alpha, sample)

    # calculate beta parameters after each update from the sample 
    beta_idx = 1
    beta_parameter_series = np.column_stack(
        (dirichlet_parameter_series[:, beta_idx],
         dirichlet_parameter_series.sum(axis=1) - 
         dirichlet_parameter_series[:, beta_idx]))
    loop_len = min(n_intervals, len(beta_parameter_series)-1)
    beta_parameters = beta_parameter_series[loop_len, :] 

    true_beta_mode = true_alpha_mode[beta_idx] 
    x, _, _ = beta_statistical_attributes()

    layout = go.Layout({
        'template': 'plotly_white',
        'showlegend': False,
        'font_size': 24})
    max_line_width = 5
    trace_color = 'red'

    fig = go.Figure(layout=layout)

    fig.add_vline(x=true_beta_mode)

    loop_len = min(n_intervals+1, len(beta_parameter_series))
    for i in range(loop_len):

        # make the most recently added lines/traces thicker than older traces
        line_width = [
            max(1, max_line_width - (loop_len - j - 1)) 
            for j in range(loop_len)]
        line01 = {'color': trace_color, 'width': line_width[i]}

        # make the most recently added lines/traces more opaque than older traces
        line_opacity = np.linspace(0.3, 1, loop_len)
        if len(line_opacity) == 1:
            line_opacity = [1]

        # calculate density for each beta distribution
        beta_parameter_alpha = beta_parameter_series[i][0]
        beta_parameter_beta = beta_parameter_series[i][1]
        y = beta.pdf(x, beta_parameter_alpha, beta_parameter_beta)

        fig.add_trace(go.Scatter(x=x, y=y, line=line01, opacity=line_opacity[i]))

    beta_mode = calculate_beta_mode(beta_parameters[0], beta_parameters[1])

    fig.add_vline(x=beta_mode, line_color=trace_color)
    beta_mode = round(beta_mode, 2)
    title = (f'B Mode ={true_beta_mode: .2f}<br><span style="color:red">'
             f'Estimate ={beta_mode: .2f}</span>')

    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='black', tick0=0, dtick=0.2,
        autorange='reversed', tickangle=90)
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
     Input('dirichlet_parameter_3', 'value'),
     Input('multinomial_proportion_1', 'value'),
     Input('multinomial_proportion_2', 'value'),
     Input('data_points_n', 'value'),
     Input('random_state', 'value')])
def plot_beta_03(
    n_intervals: int, dirichlet_parameter_1: float,
    dirichlet_parameter_2: float, dirichlet_parameter_3: float,
    multinomial_proportion_1: float, multinomial_proportion_2: float, 
    data_points_n: int, random_state: int) -> go.Figure:
    """
    Plots sequence of beta distribution updates with more recent distributions
        shown with thicker and more opaque curves
    Specifies the true, user-specified proportion of the binomial distribution
        as well as the most recent beta distribution's beta estimate (i.e., the
        mode of the distribution) of that proportion
    """

    # generate multinomial sample based on user's specified mode/proportions
    true_alpha_mode = calculate_dirichlet_mode_from_user_input(
        multinomial_proportion_1, multinomial_proportion_2)
    sample = generate_multinomial_data(
        true_alpha_mode, data_points_n, random_state)

    # calculate Dirichlet parameters after each update from the sample 
    alpha = (
        dirichlet_parameter_1, dirichlet_parameter_2, dirichlet_parameter_3)
    dirichlet_parameter_series = calculate_dirichlet_parameter_series(
        alpha, sample)

    # calculate beta parameters after each update from the sample 
    beta_idx = 2
    beta_parameter_series = np.column_stack(
        (dirichlet_parameter_series[:, beta_idx],
         dirichlet_parameter_series.sum(axis=1) - 
         dirichlet_parameter_series[:, beta_idx]))
    loop_len = min(n_intervals, len(beta_parameter_series)-1)
    beta_parameters = beta_parameter_series[loop_len, :] 

    true_beta_mode = true_alpha_mode[beta_idx] 
    x, _, _ = beta_statistical_attributes()

    layout = go.Layout({
        'template': 'plotly_white',
        'showlegend': False,
        'font_size': 24})
    max_line_width = 5
    trace_color = 'red'

    fig = go.Figure(layout=layout)

    fig.add_vline(x=true_beta_mode)

    loop_len = min(n_intervals+1, len(beta_parameter_series))
    for i in range(loop_len):

        # make the most recently added lines/traces thicker than older traces
        line_width = [
            max(1, max_line_width - (loop_len - j - 1)) 
            for j in range(loop_len)]
        line01 = {'color': trace_color, 'width': line_width[i]}

        # make the most recently added lines/traces more opaque than older traces
        line_opacity = np.linspace(0.3, 1, loop_len)
        if len(line_opacity) == 1:
            line_opacity = [1]

        # calculate density for each beta distribution
        beta_parameter_alpha = beta_parameter_series[i][0]
        beta_parameter_beta = beta_parameter_series[i][1]
        y = beta.pdf(x, beta_parameter_alpha, beta_parameter_beta)

        fig.add_trace(go.Scatter(x=x, y=y, line=line01, opacity=line_opacity[i]))

    beta_mode = calculate_beta_mode(beta_parameters[0], beta_parameters[1])

    fig.add_vline(x=beta_mode, line_color=trace_color)
    beta_mode = round(beta_mode, 2)
    title = (f'C Mode ={true_beta_mode: .2f}<br><span style="color:red">'
             f'Estimate ={beta_mode: .2f}</span>')

    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='black', tick0=0, dtick=0.2, 
        tickangle=90)
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor='black', showticklabels=False) 
    fig.update_layout(title=title, title_x=0.5)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
