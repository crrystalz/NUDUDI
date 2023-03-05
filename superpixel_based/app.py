# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import json
import io
from dash import Dash, html, dcc, Input, State, Output, ctx
import plotly.express as px
from PIL import Image
import skimage
import base64
import numpy as np
from single_image_superpixels import SingleImage
import plotly.graph_objects as go
import pickle

app = Dash(__name__)


def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y=[]))
    fig.update_layout(template=None)
    fig.update_layout(paper_bgcolor="#000")
    fig.update_layout(plot_bgcolor="#000")
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    return fig


app.layout = html.Div(
    children=[
        html.H1(
            "CADAI: Crop Anomaly Detection from Aerial Images",
            style={"textAlign": "center", "color": "#7FDBFF"},
        ),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Farm Image File")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            # Allow multiple files to be uploaded
            multiple=True,
        ),
        html.Div(id="output-data-upload"),
        html.Div(
            [
                html.Div("Number of segments in SLIC algorithm: "),
                html.Div(
                    dcc.Slider(
                        id="number-of-segments-in-slic",
                        min=200,
                        max=600,
                        step=10,
                        value=400,
                    ),
                    style={"margin": 30},
                ),
            ]
        ),
        html.Div("Maximum allowable distance between segments for clustering: "),
        html.Div(
            dcc.Slider(id="threshold", min=5, max=20, step=0.5, value=13),
            style={"margin": 30},
        ),
        html.Br(),
        html.Div(
            [
                html.Button("Compute Initial Clusters", id="compute-val", n_clicks=0),
                html.Button(
                    "Recompute Clusters",
                    id="recompute-val",
                    n_clicks=0,
                    # Add some spacing between the buttons.
                    style={"margin-left": "50px"},
                ),
                html.Button(
                    "Show superpixels",
                    id="show-superpixels",
                    n_clicks=0,
                    # Add some spacing between the buttons.
                    style={"margin-left": "50px"},
                ),
            ]
        ),
        html.Div(id="results-header"),
        dcc.Graph(id="graph", figure=blank_fig()),
        html.Div(
            [
                dcc.Markdown(
                    """
                **Regions marked non-anomalous**
            """
                ),
                html.Pre(
                    id="click-data",
                    style={
                        "border": "thin lightgrey solid",
                        "overflowX": "scroll",
                        "width": "30%",
                    },
                ),
            ],
            className="three columns",
        ),
        # Same as the local store but will lose the data
        # when the browser/tab closes.
        dcc.Store(id="clusters", storage_type="memory"),
        dcc.Store(id="coordinates", storage_type="memory"),
    ]
)

# Convert contents of uploaded file into an skimage Image


def decode_image(uploaded_file_content):
    content_type, base64_string = uploaded_file_content.split(",")
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")

    imgdata = base64.b64decode(base64_string)
    img = skimage.io.imread(imgdata, plugin="imageio")
    return img


def convert_skimage_to_pil_image(image):
    return np.array(Image.fromarray(skimage.img_as_ubyte(image)))


def make_figure(image):
    fig = px.imshow(convert_skimage_to_pil_image(image))
    fig.update_layout(showlegend=False, height=700, width=700)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(paper_bgcolor="#000")
    fig.update_layout(plot_bgcolor="#000")
    return fig


@app.callback(
    Output("results-header", "children"),
    Output("graph", "figure"),
    Output("clusters", "data"),
    Output("click-data", "children"),
    Output("coordinates", "data"),
    Input("compute-val", "n_clicks"),
    Input("recompute-val", "n_clicks"),
    Input("upload-data", "contents"),
    Input("graph", "clickData"),
    Input("show-superpixels", "n_clicks"),
    [
        State("clusters", "data"),
        State("number-of-segments-in-slic", "value"),
        State("threshold", "value"),
        State("coordinates", "data"),
    ],
)
def compute_clusters(
    compute_n_clicks,
    recompute_n_clicks,
    list_of_contents,
    graph_click_data,
    show_superpixels_n_clicks,
    clusters_data,
    num_segments,
    threshold,
    coordinates_data,
):

    # Page loaded
    if ctx.triggered_id is None:
        print("Page loaded")
        return html.H2(""), blank_fig(), "", json.dumps([]), json.dumps([])

    # Image uploaded
    if ctx.triggered_id == "upload-data":
        print("File uploaded")
        clusters = SingleImage(decode_image(list_of_contents[0]))
        return (
            html.H2("Original Image"),
            make_figure(clusters.image()),
            base64.b64encode(pickle.dumps(clusters)).decode("utf-8"),
            json.dumps([]),
            json.dumps([]),
        )

    # Compute button clicked.
    if ctx.triggered_id == "compute-val":
        print("Compute button clicked ", compute_n_clicks)
        clusters = pickle.loads(base64.b64decode(clusters_data))
        clusters.compute(threshold, num_segments)
        return (
            html.H2("Detected Regions"),
            make_figure(clusters.image_with_regions()),
            base64.b64encode(pickle.dumps(clusters)).decode("utf-8"),
            json.dumps([]),
            json.dumps([]),
        )

    # Compute button clicked.
    if ctx.triggered_id == "recompute-val":
        print("Recompute button clicked")
        clusters = pickle.loads(base64.b64decode(clusters_data))
        coordinates = json.loads(coordinates_data)
        normal_regions = []
        for c in coordinates:
            normal_regions.append((c["x"], c["y"]))
        clusters.recompute(normal_regions)
        return (
            html.H2("Detected Regions"),
            make_figure(clusters.image_with_regions()),
            base64.b64encode(pickle.dumps(clusters)).decode("utf-8"),
            json.dumps([]),
            json.dumps([]),
        )

    # Graph clicked on.
    if ctx.triggered_id == "graph":
        print("graph clicked")
        x = graph_click_data["points"][0]["x"]
        y = graph_click_data["points"][0]["y"]
        coordinates = json.loads(coordinates_data)
        coordinates.append({"x": x, "y": y})
        j = json.dumps(coordinates)
        clusters = pickle.loads(base64.b64decode(clusters_data))
        return (
            html.H2("Detected Regions"),
            make_figure(clusters.image_with_regions()),
            clusters_data,
            j,
            j,
        )

    # Show superpixels clicked.
    if ctx.triggered_id == "show-superpixels":
        print("Show superpixels clicked")
        clusters = pickle.loads(base64.b64decode(clusters_data))
        return (
            html.H2("Superpixels"),
            make_figure(clusters.image_with_superpixels()),
            base64.b64encode(pickle.dumps(clusters)).decode("utf-8"),
            graph_click_data,
            graph_click_data,
        )


if __name__ == "__main__":
    app.run_server(debug=True)
