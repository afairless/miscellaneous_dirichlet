conda create --name web_dashboard pandas numpy scikit-learn scipy matplotlib seaborn bokeh plotly dash flask streamz  altair streamlit requests jupytext scikit-image

source activate web_dashboard

conda env export > environment.yml
