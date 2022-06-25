#! usr/bin/env python3

from typing import Sequence

import numpy as np
from pandas import Series as pd_Series
from scipy.stats import beta, dirichlet

import plotly.graph_objects as go
import dash
from dash.dependencies import Input, Output
from dash_daq import BooleanSwitch


def calculate_dirichlet_parameters_from_user_input(
    A: float, B: float, C: float, not_A: int, not_B: int, not_C: int, 
    ) -> tuple[float, float, float]: 

    alpha_A = A + (not_B / 2) + (not_C / 2)
    alpha_B = B + (not_A / 2) + (not_C / 2)
    alpha_C = C + (not_A / 2) + (not_B / 2)

    return alpha_A, alpha_B, alpha_C


def calculate_dirichlet_mode(alpha: Sequence[float]) -> np.ndarray:
    """
    """

    greater_than_1 = [True if e > 1 else False for e in alpha]

    if sum(greater_than_1) == len(greater_than_1):

        alpha_sum = sum(alpha)
        alpha_len = len(alpha)
        dirichlet_mode = [(e - 1) / (alpha_sum - alpha_len) for e in alpha]

        return np.array(dirichlet_mode)

    else:

        return np.array([1/3, 1/3, 1/3])


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


# vertical white space between rows of input widgets
padding_space = "15px"

app = dash.Dash()

# layout from:  https://community.plotly.com/t/two-graphs-side-by-side/5312/2
app.layout = dash.html.Div([

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
                min=1.01, max=500, step=0.01, value=1.05, placeholder='Alpha 1'), 
            className="offset-by-two one columns"),
        dash.html.Div(
            children=dash.dcc.Input(
                id='dirichlet_parameter_2', type='number', 
                min=1.01, max=500, step=0.01, value=1.05, placeholder='Alpha 2'), 
            className="one columns"),
        dash.html.Div(
            children=dash.dcc.Input(
                id='dirichlet_parameter_3', type='number', 
                min=1.01, max=500, step=0.01, value=1.05, placeholder='Alpha 3'), 
            className="one columns"),
        ], className="row"),

    dash.html.Div(style={"padding": padding_space}),

    dash.html.Div([
        dash.html.Div(
            children=BooleanSwitch(
                id='show-betas', 
                on=True, 
                label='Show Beta Plots'), 
            className="two columns"),
        dash.html.Div(
            children=dash.html.Button(
                children='A', 
                id='data_point_A', 
                n_clicks=0), 
            className="one columns"),
        dash.html.Div(
            children=dash.html.Button(
                children='B', 
                id='data_point_B', 
                n_clicks=0), 
            className="one columns"),
        dash.html.Div(
            children=dash.html.Button(
                children='C', 
                id='data_point_C', 
                n_clicks=0), 
            className="one columns"),
        dash.html.Div(
            children=dash.html.Button(
                children='Not A', 
                id='data_point_not_A', 
                n_clicks=0), 
            className="one columns"),
        dash.html.Div(
            children=dash.html.Button(
                children='Not B', 
                id='data_point_not_B', 
                n_clicks=0), 
            className="one columns"),
        dash.html.Div(
            children=dash.html.Button(
                children='Not C', 
                id='data_point_not_C', 
                n_clicks=0), 
            className="one columns"),
    ], className="row"),


    dash.html.Div(style={"padding": padding_space}),

    dash.html.Div([dash.html.H2(id='heading1', style={'textAlign': 'center'})]),
    dash.html.Div([dash.html.H2(id='heading2', style={'textAlign': 'center'})]),

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
    Output('heading1', 'children'),
    [Input('dirichlet_parameter_1', 'value'),
     Input('dirichlet_parameter_2', 'value'),
     Input('dirichlet_parameter_3', 'value'),
     Input('data_point_A', 'n_clicks'),
     Input('data_point_B', 'n_clicks'),
     Input('data_point_C', 'n_clicks'),
     Input('data_point_not_A', 'n_clicks'),
     Input('data_point_not_B', 'n_clicks'),
     Input('data_point_not_C', 'n_clicks')])
def overall_heading1(
    prior_A: float, prior_B: float, prior_C: float, 
    A: int, B: int, C: int, 
    not_A: int, not_B: int, not_C: int) -> str: 
    """
    """

    alpha_A, alpha_B, alpha_C = calculate_dirichlet_parameters_from_user_input(
        prior_A + A, prior_B + B, prior_C + C, not_A, not_B, not_C)

    title = (#f'{prior_A}, {prior_B}, {prior_C}, {A}, {B}, {C} '
        f'Parameters: A ={alpha_A: .2f}, '
        f'B ={alpha_B: .2f}, '
        f'C ={alpha_C: .2f}')

    return title


@app.callback(
    Output('heading2', 'children'),
    [Input('dirichlet_parameter_1', 'value'),
     Input('dirichlet_parameter_2', 'value'),
     Input('dirichlet_parameter_3', 'value'),
     Input('data_point_A', 'n_clicks'),
     Input('data_point_B', 'n_clicks'),
     Input('data_point_C', 'n_clicks'),
     Input('data_point_not_A', 'n_clicks'),
     Input('data_point_not_B', 'n_clicks'),
     Input('data_point_not_C', 'n_clicks')])
def overall_heading2(
    prior_A: float, prior_B: float, prior_C: float, 
    A: int, B: int, C: int, 
    not_A: int, not_B: int, not_C: int) -> str: 
    """
    """

    alpha_A, alpha_B, alpha_C = calculate_dirichlet_parameters_from_user_input(
        prior_A + A, prior_B + B, prior_C + C, not_A, not_B, not_C)

    dirichlet_parameters = alpha_A, alpha_B, alpha_C 
    dirichlet_mode = calculate_dirichlet_mode(dirichlet_parameters)

    title = (
        f'Mode: A ={dirichlet_mode[0]: .2f}, '
        f'B ={dirichlet_mode[1]: .2f}, '
        f'C ={dirichlet_mode[2]: .2f}')

    return title


@app.callback(
    Output('dirichlet_3d', 'figure'),
    [Input('dirichlet_parameter_1', 'value'),
     Input('dirichlet_parameter_2', 'value'),
     Input('dirichlet_parameter_3', 'value'),
     Input('data_point_A', 'n_clicks'),
     Input('data_point_B', 'n_clicks'),
     Input('data_point_C', 'n_clicks'),
     Input('data_point_not_A', 'n_clicks'),
     Input('data_point_not_B', 'n_clicks'),
     Input('data_point_not_C', 'n_clicks')])
def plot_dirichlet_3d(
    prior_A: float, prior_B: float, prior_C: float, 
    A: int, B: int, C: int, 
    not_A: int, not_B: int, not_C: int) -> go.Figure: 

    alpha_A, alpha_B, alpha_C = calculate_dirichlet_parameters_from_user_input(
        prior_A + A, prior_B + B, prior_C + C, not_A, not_B, not_C)

    dirichlet_parameters = alpha_A, alpha_B, alpha_C 
    dirichlet_mode = calculate_dirichlet_mode(dirichlet_parameters)
    grid = create_barycentric_grid_coordinates(91)

    # apply probability density function across triangular plot grid
    pdf = pd_Series([dirichlet.pdf(e, dirichlet_parameters) for e in grid])
    pdf_adjusted = pdf.copy()

    max_idx = np.where(grid==grid.max())[0]
    for idx in max_idx:
        pdf_adjusted.iat[idx] = pdf_adjusted.max()
    
    grid_max_idx = grid.argmax(axis=1)
    
    grids = []
    pdfs = []
    pdfs_adjusted = []
    for i in range(3):
        grids.append(grid[grid_max_idx == i, :])
        pdfs.append(pdf[grid_max_idx == i])
        pdfs_adjusted.append(pdf_adjusted[grid_max_idx == i])

    scatter_point_size = 8
    scatter_point_opacity = 0.9

    fig_trace01 = go.Scatterternary({
        'mode': 'markers',
        'a': grids[0][:, 0],
        'b': grids[0][:, 1],
        'c': grids[0][:, 2],
        'marker': {
            #'symbol': 100,
            'color': pdfs_adjusted[0],
            'colorscale': 'blues',
            'size': scatter_point_size,
            'opacity': scatter_point_opacity}
    })

    fig_trace02 = go.Scatterternary({
        'mode': 'markers',
        'a': grids[1][:, 0],
        'b': grids[1][:, 1],
        'c': grids[1][:, 2],
        'marker': {
            #'symbol': 100,
            'color': pdfs_adjusted[1],
            'colorscale': 'reds',
            'size': scatter_point_size,
            'opacity': scatter_point_opacity}
    })

    fig_trace03 = go.Scatterternary({
        'mode': 'markers',
        'a': grids[2][:, 0],
        'b': grids[2][:, 1],
        'c': grids[2][:, 2],
        'marker': {
            #'symbol': 100,
            'color': pdfs_adjusted[2],
            'colorscale': 'greens',
            'size': scatter_point_size,
            'opacity': scatter_point_opacity}
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
            'opacity': 0.7}
    })

    layout01 = {
        'template': 'plotly_white',
        'showlegend': False}

    title_font_size = 40
    tick_font_size = 15
    tick_length = 2

    layout02 = {
        'ternary': {
            'sum': 1,
            'aaxis': {
                'title': 'A', 'titlefont': {'size': title_font_size}, 
                'tickangle': 0, 'tickfont': {'size': tick_font_size}, 
                'tickcolor': 'rgba(0,0,0,0)', 'color': 'blue',
                'ticklen': tick_length, 'showline': True, 'showgrid': True},
            'baxis': {
                'title': 'B', 'titlefont': {'size': title_font_size}, 
                'tickangle': 45, 'tickfont': {'size': tick_font_size}, 
                'tickcolor': 'rgba(0,0,0,0)', 'color': 'red',
                'ticklen': tick_length, 'showline': True, 'showgrid': True},
            'caxis': {
                'title': 'C', 'titlefont': {'size': title_font_size}, 
                'tickangle': -45, 'tickfont': {'size': tick_font_size}, 
                'tickcolor': 'rgba(0,0,0,0)', 'color': 'green',
                'ticklen': tick_length, 'showline': True, 'showgrid': True},
        },
    }

    fig = go.Figure(data=fig_trace01, layout=layout01)
    fig.add_trace(fig_trace02)
    fig.add_trace(fig_trace03)
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
    [Input('dirichlet_parameter_1', 'value'),
     Input('dirichlet_parameter_2', 'value'),
     Input('dirichlet_parameter_3', 'value'),
     Input('data_point_A', 'n_clicks'),
     Input('data_point_B', 'n_clicks'),
     Input('data_point_C', 'n_clicks'),
     Input('data_point_not_A', 'n_clicks'),
     Input('data_point_not_B', 'n_clicks'),
     Input('data_point_not_C', 'n_clicks')])
def plot_beta_01(
    prior_A: float, prior_B: float, prior_C: float, 
    A: int, B: int, C: int, 
    not_A: int, not_B: int, not_C: int) -> go.Figure: 
    """
    Plots sequence of beta distribution updates with more recent distributions
        shown with thicker and more opaque curves
    Specifies the true, user-specified proportion of the binomial distribution
        as well as the most recent beta distribution's beta estimate (i.e., the
        mode of the distribution) of that proportion
    """

    alpha_A, alpha_B, alpha_C = calculate_dirichlet_parameters_from_user_input(
        prior_A + A, prior_B + B, prior_C + C, not_A, not_B, not_C)

    # calculate beta parameters after each update from the sample 
    beta_parameter_alpha = alpha_A
    beta_parameter_beta = alpha_B + alpha_C

    x, _, _ = beta_statistical_attributes()

    layout = go.Layout({
        'template': 'plotly_white',
        'showlegend': False,
        'font_size': 24})
    trace_color = 'red'

    fig = go.Figure(layout=layout)

    line01 = {'color': trace_color, 'width': 5}
    y = beta.pdf(x, beta_parameter_alpha, beta_parameter_beta)
    fig.add_trace(go.Scatter(x=x, y=y, line=line01))

    beta_mode = calculate_beta_mode(beta_parameter_alpha, beta_parameter_beta)
    fig.add_vline(x=beta_mode, line_color=trace_color)

    beta_mode = round(beta_mode, 2)
    title = (f'A<br><span style="color:red">Mode ={beta_mode: .2f}</span>')

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
    [Input('dirichlet_parameter_1', 'value'),
     Input('dirichlet_parameter_2', 'value'),
     Input('dirichlet_parameter_3', 'value'),
     Input('data_point_A', 'n_clicks'),
     Input('data_point_B', 'n_clicks'),
     Input('data_point_C', 'n_clicks'),
     Input('data_point_not_A', 'n_clicks'),
     Input('data_point_not_B', 'n_clicks'),
     Input('data_point_not_C', 'n_clicks')])
def plot_beta_02(
    prior_A: float, prior_B: float, prior_C: float, 
    A: int, B: int, C: int, 
    not_A: int, not_B: int, not_C: int) -> go.Figure: 
    """
    Plots sequence of beta distribution updates with more recent distributions
        shown with thicker and more opaque curves
    Specifies the true, user-specified proportion of the binomial distribution
        as well as the most recent beta distribution's beta estimate (i.e., the
        mode of the distribution) of that proportion
    """

    alpha_A, alpha_B, alpha_C = calculate_dirichlet_parameters_from_user_input(
        prior_A + A, prior_B + B, prior_C + C, not_A, not_B, not_C)

    # calculate beta parameters after each update from the sample 
    beta_parameter_alpha = alpha_B
    beta_parameter_beta = alpha_A + alpha_C

    x, _, _ = beta_statistical_attributes()

    layout = go.Layout({
        'template': 'plotly_white',
        'showlegend': False,
        'font_size': 24})
    trace_color = 'red'

    fig = go.Figure(layout=layout)

    line01 = {'color': trace_color, 'width': 5}
    y = beta.pdf(x, beta_parameter_alpha, beta_parameter_beta)
    fig.add_trace(go.Scatter(x=x, y=y, line=line01))

    beta_mode = calculate_beta_mode(beta_parameter_alpha, beta_parameter_beta)
    fig.add_vline(x=beta_mode, line_color=trace_color)

    beta_mode = round(beta_mode, 2)
    title = (f'B<br><span style="color:red">Mode ={beta_mode: .2f}</span>')

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
    [Input('dirichlet_parameter_1', 'value'),
     Input('dirichlet_parameter_2', 'value'),
     Input('dirichlet_parameter_3', 'value'),
     Input('data_point_A', 'n_clicks'),
     Input('data_point_B', 'n_clicks'),
     Input('data_point_C', 'n_clicks'),
     Input('data_point_not_A', 'n_clicks'),
     Input('data_point_not_B', 'n_clicks'),
     Input('data_point_not_C', 'n_clicks')])
def plot_beta_03(
    prior_A: float, prior_B: float, prior_C: float, 
    A: int, B: int, C: int, 
    not_A: int, not_B: int, not_C: int) -> go.Figure: 
    """
    Plots sequence of beta distribution updates with more recent distributions
        shown with thicker and more opaque curves
    Specifies the true, user-specified proportion of the binomial distribution
        as well as the most recent beta distribution's beta estimate (i.e., the
        mode of the distribution) of that proportion
    """

    alpha_A, alpha_B, alpha_C = calculate_dirichlet_parameters_from_user_input(
        prior_A + A, prior_B + B, prior_C + C, not_A, not_B, not_C)

    # calculate beta parameters after each update from the sample 
    beta_parameter_alpha = alpha_C
    beta_parameter_beta = alpha_A + alpha_B

    x, _, _ = beta_statistical_attributes()

    layout = go.Layout({
        'template': 'plotly_white',
        'showlegend': False,
        'font_size': 24})
    trace_color = 'red'

    fig = go.Figure(layout=layout)

    line01 = {'color': trace_color, 'width': 5}
    y = beta.pdf(x, beta_parameter_alpha, beta_parameter_beta)
    fig.add_trace(go.Scatter(x=x, y=y, line=line01))

    beta_mode = calculate_beta_mode(beta_parameter_alpha, beta_parameter_beta)
    fig.add_vline(x=beta_mode, line_color=trace_color)

    beta_mode = round(beta_mode, 2)
    title = (f'C<br><span style="color:red">Mode ={beta_mode: .2f}</span>')

    fig.update_xaxes(
        showline=True, linewidth=2, linecolor='black', tick0=0, dtick=0.2, 
        tickangle=90)
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor='black', showticklabels=False)
    fig.update_layout(title=title, title_x=0.5)

    return fig


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
