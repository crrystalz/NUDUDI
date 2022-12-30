import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from PIL import Image

#
# Given a list of rectangles on X/Y coordinate plan, find the convex hull,
# which is the smallest polygon that encloses all rectangles within it.
#

def minimum_bounding_polygon(rectangles):
    # create a list of polygons from the rectangles
    polygons = [Polygon([(r[0], r[1]), (r[0], r[3]), (r[2], r[3]), (r[2], r[1])]) for r in rectangles]

    # create a multipolygon from the polygons
    multipoly = MultiPolygon(polygons)

    # return the convex hull of the multipolygon
    return multipoly.convex_hull


def draw_polygon_on_image(polygons, image) -> None:
    # Convert the image to a NumPy array
    img_array = np.array(image)

    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(img_array)

    for polygon in polygons:
        # Extract the coordinates of the polygon
        x, y = polygon.exterior.xy

        # Plot the polygon on top of the image
        ax.plot(x, y)

    # Show the plot
    plt.show()


def x(clusters, image_file):
    polygons = []
    for cluster in clusters:
        cell_rectangles = []
        for cell in cluster.cells:
            cell_rectangle = cell.get_rectangle()
            cell_rectangles.append(cell_rectangle)

        
        polygons.append(minimum_bounding_polygon(cell_rectangles))

    draw_polygon_on_image(polygons, image_file)

        