import os
from PIL import Image
import numpy as np
import shutil
import cv2
import matplotlib.pyplot as plt

from utils import printProgressBar

# Type 1


def combine_files(directory):
    for field_id in os.listdir(directory):
        print("New image: " + field_id)
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


# Type 2


def is_black(image, tolerance=10):
    # Check if all pixels are within the tolerance of black (0)
    return np.all(np.abs(image - 0) <= tolerance)


def create_dataset(
    path_to_train_dir, output_dataset_dir, num_anomalies, images_per_anomaly=-1
):
    for root, dirs, files in os.walk(output_dataset_dir):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

    print("Deleted previous files")

    images_dir = os.path.join(path_to_train_dir, "images", "rgb")
    labels_dir = os.path.join(path_to_train_dir, "labels")

    rgb_images_path = path_to_train_dir + "\\images\\rgb"
    total_files = sum([len(files) for _, _, files in os.walk(rgb_images_path)])
    num_files = 0

    if images_per_anomaly == -1:
        images_per_anomaly = total_files
        num_files = total_files * num_anomalies
    else:
        num_files = images_per_anomaly * num_anomalies

    print()
    printProgressBar(0, num_files, prefix="Progress:", suffix="Complete", length=50)
    progress = 0

    fig, ax = plt.subplots()

    for anomaly_type in os.listdir(labels_dir):
        anomaly_dir = os.path.join(labels_dir, anomaly_type)
        anomaly_dir_files = os.listdir(anomaly_dir)

        # For each anomaly type, create a directory stucture
        # like the following inside output_dataset_dir.
        #
        # output_dataset_dir
        #                   |-- anomaly_type
        #                                  |-- image
        #                                  |-- annotated_image
        output_anomaly_dir = os.path.join(output_dataset_dir, anomaly_type)
        if not os.path.exists(output_anomaly_dir):
            os.mkdir(output_anomaly_dir)

        output_image_dir = os.path.join(output_anomaly_dir, "image")
        if not os.path.exists(output_image_dir):
            os.mkdir(output_image_dir)

        output_annotated_image_dir = os.path.join(output_anomaly_dir, "annotated-image")
        if not os.path.exists(output_annotated_image_dir):
            os.mkdir(output_annotated_image_dir)

        count = 0
        i = 0
        while count != images_per_anomaly:
            file_name = anomaly_dir_files[i]

            image = cv2.imread(f"{anomaly_dir}\\{file_name}", cv2.IMREAD_GRAYSCALE)
            if not is_black(image):
                if os.path.exists(f"{images_dir}\\{file_name}"):
                    # copy the image file into the output dir
                    shutil.copy(
                        f"{images_dir}\\{file_name}", f"{output_image_dir}\\{file_name}"
                    )
                else:
                    shutil.copy(
                        str(images_dir) + "\\" + file_name[:-3] + "jpg",
                        f"{output_image_dir}\\{file_name}",
                    )

                anomalous_image_bw = Image.open(f"{anomaly_dir}\\{file_name}").convert(
                    "L"
                )
                anomalous_image_rgb = Image.open(f"{output_image_dir}\\{file_name}")
                overlay_anomalous_region_boundary(
                    anomalous_image_bw,
                    anomalous_image_rgb,
                    f"{output_annotated_image_dir}/{file_name}",
                    fig,
                    ax,
                )

                count = count + 1
                progress += 1

                if progress < num_files:
                    printProgressBar(
                        progress + 1,
                        num_files,
                        prefix="Progress:",
                        suffix="Complete",
                        length=50,
                    )

            i += 1

        plt.close(fig)

        # print(anomaly_type + " count: " + str(count))


# returns a list (x,y) corrdinates that define the boundary of
# anomalous region.
def get_anomaly_outline(anomalous_image):
    outline_x = []
    outline_y = []
    for x in range(1, anomalous_image.width - 1):
        for y in range(1, anomalous_image.height - 1):
            # Check if the pixel is white and at least one of its neighbors is black
            if anomalous_image.getpixel((x, y)) == 255 and (
                anomalous_image.getpixel((x - 1, y)) == 0
                or anomalous_image.getpixel((x + 1, y)) == 0
                or anomalous_image.getpixel((x, y - 1)) == 0
                or anomalous_image.getpixel((x, y + 1)) == 0
            ):
                outline_x.append(x)
                outline_y.append(y)
    return outline_x, outline_y


# Overlays the anomalous region outline on the image.
def overlay_anomalous_region_boundary(
    anomalous_image_bw, anomalous_image_rgb, annotated_image_file_path, fig, ax
):
    x, y = get_anomaly_outline(anomalous_image_bw)
    img_array = np.array(anomalous_image_rgb)

    ax.imshow(img_array)
    ax.scatter(x, y, s=1)
    plt.savefig(annotated_image_file_path)
    plt.clf()


# combine_files("F:/nududi_datasets/testing-2")

train_dir = "F:\\nududi_datasets\\agriculture-vision-2021-supervised\\supervised\\Agriculture-Vision-2021\\train"
output_dataset_dir = "F:\\nududi_datasets\\testing-1"
create_dataset(train_dir, output_dataset_dir, 9, 10)
