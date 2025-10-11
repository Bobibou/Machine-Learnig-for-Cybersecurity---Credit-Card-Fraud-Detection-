
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.offline import iplot, init_notebook_mode
import matplotlib.pyplot as plt
import seaborn as sns

# Enable Plotly in offline mode for Jupyter
init_notebook_mode(connected=True)

def plot_class_distribution(df, class_col="Class"):

    if class_col not in df.columns:
        raise KeyError(f"The column '{class_col}' does not exist in the DataFrame.")

    temp = df[class_col].value_counts().sort_index()
    temp_df = pd.DataFrame({'Class': temp.index, 'Values': temp.values})

    trace = go.Bar(
        x=temp_df['Class'],
        y=temp_df['Values'],
        name="Class distribution (0 = Normal, 1 = Fraud)",
        marker=dict(color=['#1f77b4', '#d62728']),
        text=temp_df['Values'],
        textposition='auto'
    )

    layout = dict(
        title='Transaction Distribution: Fraud vs Normal',
        xaxis=dict(title='Class (0 = Normal, 1 = Fraud)', showticklabels=True),
        yaxis=dict(title='Number of Transactions'),
        hovermode='closest',
        width=600,
        template='plotly_white'
    )

    fig = dict(data=[trace], layout=layout)
    iplot(fig)


