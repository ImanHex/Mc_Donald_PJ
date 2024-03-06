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

app = Dash(__name__)
server = app.server
stored_folder = Path(os.path.abspath('')) / "data" / "processed" / "cleaned_df.pkl"
input_file = open(stored_folder, "rb")
df = pickle.load(input_file)

stored_model = Path(os.path.abspath('')) / "data" / "modeling" / "cluster_models.pkl"
cluster_model = open(stored_model, "rb")
model = pickle.load(cluster_model)

unique_coordinates = df[['latitude', 'longitude', 'store_address']].drop_duplicates()

average_rating_by_store = {address: (df[df['store_address'] == address])['rating'].mean().round(2) for address in
                           df['store_address'].unique()}
print(average_rating_by_store)

average_rating_by_latitude = df.groupby('latitude')['rating'].mean().round(2)

coordinates = [(latitude, longitude) for latitude, longitude in zip(df.latitude, df.longitude)]

latitudes, longitudes = np.array(coordinates).T
print(latitudes)
print(longitudes)



# latitudes, longitudes = np.array(coordinates).T
# Define color scale for ratings
# def color_fn(rating):
#     if rating < 2.5:
#         return 'red'  # Low rating
#     elif rating < 4:
#         return 'yellow'  # Medium rating
#     else:
#         return 'green'  # High rating
#
# markers = [
#     dl.Marker(
#         position=[row['latitude'], row['longitude']],
#         children=[
#             dl.Tooltip(content="{row['store_address']} (Rating: {row['rating']:.2f})")
#         ],
#         style={
#             'color': 'black',
#             'backgroundColor': color_fn(row['rating']),
#             'radius': 10,
#             'fillOpacity': 0.7
#         }
#     ) for _, row in average_ratings.iterrows()
# ]


def cluster_to_wordcloud(cluster_group, max_words=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(cluster_group)
    vocab_list = vectorizer.get_feature_names_out()

    # create a rank list of words
    df_ranked = pd.DataFrame({'Word': vocab_list, 'Sum TFIDF': tfidf_matrix.toarray().sum(axis=0)}).sort_values(
        'Sum TFIDF', ascending=False)

    # create word score
    word_to_score = {word: score for word, score in df_ranked[:max_words].values}

    # initialize wordcloud object
    wordcloud_generator = WordCloud(background_color='white')

    # fit wordcloud_generator to word_to_score
    wordcloud_image = wordcloud_generator.fit_words(word_to_score)

    return wordcloud_image


def generate_wordcloud_image(cluster_num, index):
    cluster_group = model[index][
        cluster_num].Review  # Get cluster group data for the specified cluster number and index
    wordcloud_image = cluster_to_wordcloud(cluster_group)  # Generate word cloud image as dictionary of word frequencies
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(wordcloud_image, interpolation='bilinear')
    ax.axis("off")  # Hide axis
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)
    img_str = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{img_str}"


# Define Dash layout
app.layout = html.Div([
    html.H1("McDonald's Reviews Word Cloud"),
    html.Div(id='cluster-label', style={'marginTop': '10px'}),
    html.Div(id='index-label', style={'marginTop': '10px'}),
    html.Div(id="wordcloud-container", children=[
        html.Img(id="wordcloud-img", src=generate_wordcloud_image(0, 0)),
        # Initial word cloud image with cluster 0 and index 0
        html.Div([
            html.Label("Select Cluster Number:"),
            dcc.Dropdown(
                id='cluster-dropdown',
                options=[{'label': f'Cluster {i + 1}', 'value': i} for i in range(len(model[0]))],
                # Create dropdown options dynamically based on number of clusters
                value=0,  # Set default value to the first cluster
                clearable=False  # Disable clearing the selection
            )
        ]),
        html.Div([
            html.Label("Select Index:"),
            dcc.Dropdown(
                id='index-dropdown',
                options=[{'label': f'Rating {i + 1}', 'value': i} for i in range(len(model))],
                # Create dropdown options dynamically based on number of indexes
                value=0,  # Set default value to the first index
                clearable=False  # Disable clearing the selection
            )
        ]),

    ], style={'width': '50%', 'display': 'inline-flex', 'margin': '10px'}),
    html.Div([
        html.H1("Leaflet Map"),
        dl.Map([
            dl.TileLayer(),
            *[dl.Marker(position=[latitude, longitude], children=[
                dl.Tooltip(html.Div([
                    html.P(f"Store Address: {store_address}"),
                    html.P(f"Average Rating: {average_rating_by_latitude[latitude]}")
                ]))
            ]) for latitude, longitude, store_address in
              zip(unique_coordinates['latitude'], unique_coordinates['longitude'], unique_coordinates['store_address'])]
        ], center=[unique_coordinates['latitude'][0], unique_coordinates['longitude'][0]], zoom=4,
            style={'height': '500px', 'width': '500px'})
    ])
])


# Define callback to update word cloud image based on selected cluster number and index
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


if __name__ == "__main__":
    app.run_server(debug=True)
