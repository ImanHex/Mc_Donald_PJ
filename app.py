import os
import pickle
import numpy as np
import pandas as pd
from wordcloud import WordCloud
from io import BytesIO
import base64
from dash.dependencies import Input, Output
import dash_leaflet as dl
from dash import Dash, html, dcc
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objs as go

app = Dash(__name__)
server = app.server
stored_folder = Path(os.path.abspath('')) / "data" / "processed" / "cleaned_df.pkl"
input_file = open(stored_folder, "rb")
df = pickle.load(input_file)

stored_model = Path(os.path.abspath('')) / "data" / "modeling" / "cluster_models.pkl"
cluster_model = open(stored_model, "rb")
model = pickle.load(cluster_model)

unique_coordinates = df[['latitude', 'longitude', 'store_address']].drop_duplicates()
average_rating_by_store = df.groupby('store_address')['rating'].mean().round(2).reset_index(name='rating')

average_rating_by_latitude = df.groupby('latitude')['rating'].mean().round(2)

coordinates = [(latitude, longitude) for latitude, longitude in zip(df.latitude, df.longitude)]

latitudes, longitudes = np.array(coordinates).T


def cluster_to_wordcloud(cluster_group, max_words=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(cluster_group)
    vocab_list = vectorizer.get_feature_names_out()
    df_ranked = pd.DataFrame({'Word': vocab_list, 'Sum TFIDF': tfidf_matrix.toarray().sum(axis=0)}).sort_values(
        'Sum TFIDF', ascending=False)
    word_to_score = {word: score for word, score in df_ranked[:max_words].values}
    wordcloud_generator = WordCloud(background_color='white')
    wordcloud_image = wordcloud_generator.fit_words(word_to_score)
    return wordcloud_image


def generate_wordcloud_image(cluster_num, index):
    cluster_group = model[index][
        cluster_num].Review
    wordcloud_image = cluster_to_wordcloud(cluster_group)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(wordcloud_image, interpolation='bilinear')
    ax.axis("off")  # Hide axis
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)
    img_str = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{img_str}"


wordcloud_layout = html.Div([
    html.H1("McDonald's Reviews Word Cloud",
            style={'textAlign': 'center', 'backgroundColor': '#0181AD', 'color': 'white', 'padding': '20px',
                   'borderRadius': '5px'}),
    html.Div(id="wordcloud-container", children=[

        html.Div([
            html.Div([
                html.Label("Select Cluster Number:", style={'marginBottom': '10px', 'width': '100%'}),
                dcc.Dropdown(
                    id='cluster-dropdown',
                    options=[{'label': f'Cluster {i + 1}', 'value': i} for i in range(len(model[0]))],
                    value=0,
                    clearable=False,
                    style={'width': '100%'}
                ),
                html.Div(id='cluster-label', style={'marginTop': '10px', 'width': '100%'}),
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'flex-end', 'width': '300px',
                      'backgroundColor': '#9EE5FA', 'padding': '10px', 'borderRadius': '5px'}),

            html.Div([
                html.Label("Select Index:", style={'marginBottom': '10px', 'width': '100%'}),
                dcc.Dropdown(
                    id='index-dropdown',
                    options=[{'label': f'Rating {i + 1}', 'value': i} for i in range(len(model))],
                    value=0,
                    clearable=False,
                    style={'width': '100%'}
                ),
                html.Div(id='index-label', style={'marginTop': '10px', 'width': '100%'}),
            ], style={'display': 'flex', 'flexDirection': 'column', 'marginTop': '50px', 'alignItems': 'flex-end',
                      'width': '300px', 'backgroundColor': '#9EE5FA', 'padding': '10px', 'borderRadius': '5px'}),
        ], style={'backgroundColor': 'white', 'padding': '135px', 'alignItems': 'center'}),

        html.Img(id="wordcloud-img", src=generate_wordcloud_image(0, 0),
                 style={'backgroundColor': '#E3F6FB', 'borderRadius': '10px', 'marginRight': '5px', 'width': '50%',
                        'height': '50%'}),

    ], style={'display': 'flex', 'justifyContent': 'center', 'alignItems': 'center', 'gap': '50px',
              'borderRadius': '5px', 'backgroundColor': 'white'}),
    html.Div([
        html.H1('Number of Reviews Over Time',
                style={'textAlign': 'center', 'backgroundColor': '#0181AD', 'color': 'white', 'padding': '20px',
                       'borderRadius': '5px'}),
        dcc.Graph(id='time-series-chart'),
    ])
])

pie_chart_layout = html.Div([
    html.H1("Distribution of Review Scores", style={'textAlign': 'center', 'backgroundColor': '#0181AD', 'color': 'white', 'padding': '20px',
                   'borderRadius': '5px'}),
    dcc.Graph(id='pie-chart'),
])


map_layout = html.Div([
    pie_chart_layout,
    html.H1("Branches Average Score Map",
            style={'textAlign': 'center', 'backgroundColor': '#0181AD', 'color': 'white', 'padding': '20px',
                   'borderRadius': '5px'}),

    # Map
    dl.Map([
        dl.TileLayer(),
        *[dl.Marker(position=[latitude, longitude], children=[
            dl.Tooltip(html.Div([
                html.P(f"Store Address: {store_address}"),
                html.P(f"Average Rating: {average_rating:.2f}")
            ]))
        ]) for latitude, longitude, store_address, average_rating in
          zip(unique_coordinates['latitude'], unique_coordinates['longitude'],
              unique_coordinates['store_address'], average_rating_by_store['rating'])]
    ], center=[unique_coordinates['latitude'].mean(), unique_coordinates['longitude'].mean()], zoom=4,
        style={'height': '800px', 'width': '800px', 'margin': 'auto', 'borderRadius': '10px'}),
    html.Div([
        html.H1("Average Rating by Store Chart",
                style={'textAlign': 'center', 'backgroundColor': '#0181AD', 'color': 'white', 'padding': '20px',
                       'borderRadius': '5px'}),
        dcc.Graph(id='average-rating-bar-chart')], )

])

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.Div([
            dcc.Link('Word Cloud', href='/', style={'font-family': 'Times New Roman, Times, serif', 'font-weight': 'bold', 'marginRight': '10px'}),
            dcc.Link('Map', href='/map', style={'font-family': 'Times New Roman, Times, serif', 'font-weight': 'bold', 'marginRight': '10px'}),
        ], style={'padding': '10px'})
    ], style={'backgroundColor': '#F4F7F7', 'width': '100%'}),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return wordcloud_layout
    elif pathname == '/map':
        return map_layout
    else:
        return wordcloud_layout


@app.callback(
    [Output('wordcloud-img', 'src'),
     Output('cluster-label', 'children'),
     Output('index-label', 'children')],
    [Input('cluster-dropdown', 'value'),
     Input('index-dropdown', 'value')]
)
def update_wordcloud(cluster_num, index):
    cluster_label = f"Current Cluster: {cluster_num + 1}"
    index_label = f"Current Rating: {index + 1}"
    return generate_wordcloud_image(cluster_num, index), cluster_label, index_label


@app.callback(Output('average-rating-bar-chart', 'figure'), [Input('url', 'pathname')])
def update_bar_chart(pathname):
    data = [
        go.Bar(
            x=average_rating_by_store['store_address'],
            y=average_rating_by_store['rating'],
            marker=dict(color='rgb(26, 118, 255)')
        )
    ]
    layout = go.Layout(
        xaxis=dict(title='Store Address'),
        yaxis=dict(title='Average Rating')
    )
    return {'data': data, 'layout': layout}


@app.callback(Output("time-series-chart", "figure"), [Input("url", "pathname")])
def display_time_graph(pathname):
    review_counts = df.groupby('review_time').size().reset_index(name='Review Count')
    fig = px.line(review_counts, x='review_time', y='Review Count')
    return fig


@app.callback(
    Output('pie-chart', 'figure'),
    [Input('url', 'pathname')]
)
def update_pie_chart(pathname):
    if pathname == '/map':
        # Calculate the count of each review score
        rating_counts = df['rating'].value_counts().reset_index()

        # Rename the columns to match the expected column names
        rating_counts.columns = ['rating_score', 'total_people_votes']

        # Create the pie chart using Plotly Express
        fig = px.pie(rating_counts, values='total_people_votes', names='rating_score')

        return fig


if __name__ == "__main__":
    app.run_server(debug=True)
