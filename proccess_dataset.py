import os
from PIL import Image

# directory should be the full path to the downloaded "raw" directory
def combine_files(directory):
    for field_id in os.listdir(directory):
        combine_files_for_one_field(directory, field_id)


def combine_files_for_one_field(directory, field_id):
    # Construct the path to the subdirectory
    subdirectory_path = os.path.join(directory, field_id)

    # Check if the subdirectory exists
    if not os.path.exists(subdirectory_path):
        print(f"Error: {subdirectory_path} does not exist.")
        return

    flight_ids = set()
    for filename in os.listdir(subdirectory_path):
        if filename == "boundary.tif":
            continue

        flight_ids.add(filename.split("_")[0])

    for flight_id in flight_ids:
        # Open the red, green, and blue channel images
        red_image = Image.open(
            f"{directory}/{field_id}/{flight_id}_red_high.tif"
        ).convert("L")
        green_image = Image.open(
            f"{directory}/{field_id}/{flight_id}_green_high.tif"
        ).convert("L")
        blue_image = Image.open(
            f"{directory}/{field_id}/{flight_id}_blue_high.tif"
        ).convert("L")

        # Combine the images into a single image
        rgb_image = Image.merge("RGB", (red_image, green_image, blue_image))

        # Save the combined image
        rgb_image.save(f"{directory}/{field_id}/{flight_id}_rgb.tif")


combine_files("E:/nududi_datasets/testing2")
