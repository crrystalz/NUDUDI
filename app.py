# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
 
from dash import Dash, html, dcc, Input, State, Output
import plotly.express as px
from PIL import Image
import skimage
import superpixel_standalone
import base64
 
def convert_skimage_to_pil_image(image):
    return Image.fromarray(skimage.img_as_ubyte(image))
 
app = Dash(__name__)
 
app.layout = html.Div(children=[
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
 
    html.Div([
        "Number of segments: ",
        dcc.Input(id='number-of-segments-in-slic', value=400, type='number')
    ]),
 
    html.Div([
        "Maximum allowable distance between segments for clustering: ",
        dcc.Input(id='threshold', value=10, type='number')
    ]),
 
    #dcc.Checklist(
    #['Show original image', 'Show superpixels'],
    #['Show original image', 'Show superpixels'],
    #    inline=True,
    #    id='checklist',
    #),
 
    html.Button('Compute', id='compute-val', n_clicks=0),
 
    html.H1(children='Anomalous Regions'),
 
    html.Div(id='results-container',
             children='Enter a value and press compute')
 
])
 
def decode(base64_string):
    if isinstance(base64_string, bytes):
        base64_string = base64_string.decode("utf-8")
 
    imgdata = base64.b64decode(base64_string)
    img = skimage.io.imread(imgdata, plugin='imageio')
    return img
 
def superpixel_standalone_compute(uploaded_file_content, num_segments, threshold):
    content_type, base64_encoded_image_data = uploaded_file_content.split(',')
    img = decode(base64_encoded_image_data)
    superpixel_img, merged_superpixel_img = superpixel_standalone.compute(img, threshold, num_segments)
    return html.Tr(
            [html.Td(html.Img(src=convert_skimage_to_pil_image(img))),
             html.Td(html.Img(src=convert_skimage_to_pil_image(superpixel_img))),
             html.Td(html.Img(src=convert_skimage_to_pil_image(merged_superpixel_img))),
            ])
 
@app.callback(
    Output(component_id='results-container', component_property='children'),
    Input('compute-val', 'n_clicks'),
    [State('upload-data', 'contents'),
     State('number-of-segments-in-slic', 'value'),
     State('threshold', 'value'),
     #State('checklist', 'value'),
     ]
)
def update_output_div(n_clicks, list_of_contents, num_segments, threshold):
   if list_of_contents is None:
       return html.Table([])
   return html.Table(
       [html.Tr([html.Td(html.H2("Original Image")),
                 html.Td(html.H2("After superpixel transformation")),
                 html.Td(html.H2("After clustering"))])] +
       [superpixel_standalone_compute(c, num_segments, threshold) for c in list_of_contents])
 
if __name__ == '__main__':
    app.run_server(debug=True)
 