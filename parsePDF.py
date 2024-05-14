import os
import re
import sys
import time
import numpy as np

# Coumpute vision
import cv2

# Detect text from image
import pytesseract

# Convert pdf file to image
from pdf2image import convert_from_path


input_folder = "uploads"
output_folder = "output"
tessdata_dir_config = '--tessdata-dir "traineddata"'
name_table = "table_with_lines"
skip_page = ["page_1.jpg", name_table + ".jpg"]
allowed_formats = [".pdf"]

save_to_file = False
display_logs = True
save_table = True


def log(text):
    if display_logs == True:
        print(text)


def clear_folder():
    # Ensure output folder exists, create it if not
    if not os.path.exists(output_folder):
        return os.makedirs(output_folder)

    # List all files in the folder
    files = os.listdir(output_folder)

    # Iterate over each file
    for file_name in files:
        # Construct the full path to the file
        file_path = os.path.join(output_folder, file_name)

        # Check if the path points to a file
        if os.path.isfile(file_path):
            # Remove the file
            os.remove(file_path)

    log(f"Clean all file in {output_folder} folder")


def convert_pdf_to_image(pdf):
    # Convert PDF to list of PIL.Image
    images = convert_from_path(pdf)

    # Iterate through the images and save them
    for i, image in enumerate(images):
        image.save(f"{output_folder}/page_{i+1}.jpg", "JPEG")

    log("Convert pdf to image.")


def find_table_in_image(image):
    # Load the imag
    image = cv2.imread(f"{output_folder}/page_1.jpg")

    # Convert image to Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding to prepare for table detection
    _, bin_image = cv2.threshold(
        gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # Inverting the image
    bin_image = 255 - bin_image

    # Detect lines (highlighted lines)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horizontal_lines = cv2.morphologyEx(
        bin_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    vertical_lines = cv2.morphologyEx(
        bin_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )

    # Apply dilation to make lines thicker
    kernel = np.ones((2, 2), np.uint8)
    horizontal_lines = cv2.dilate(horizontal_lines, kernel, iterations=1)
    vertical_lines = cv2.dilate(vertical_lines, kernel, iterations=1)

    # Combine lines to create table structure
    table_structure = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # Find contours for table detection
    contours, _ = cv2.findContours(
        table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding rectangle of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the cell from the original image
    cell_image = image[y : y + h, x : x + w]

    # Horizonal and vertical line in table
    find_table_h_l = horizontal_lines[y : y + h, x : x + w] > 0
    find_table_w_l = vertical_lines[y : y + h, x : x + w] > 0

    # Draw highlighted lines on the cell image
    cell_with_lines = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    cell_with_lines = cv2.cvtColor(cell_with_lines, cv2.COLOR_GRAY2BGR)
    cell_with_lines[find_table_h_l] = [0, 0, 255]  # Red color for horizontal lines
    cell_with_lines[find_table_w_l] = [0, 0, 255]  # Red color for vertical lines

    if save_table == True:
        # Save the cell with highlighted lines as a new image
        cv2.imwrite(f"{output_folder}/{name_table}.jpg", cell_with_lines)
    # show_image(cell_with_lines)

    log("Find table in image.")
    return cell_with_lines


def find_horizonal_lines(img, minLineLength=100):
    # Pretvorba u sivu skalu
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding to prepare for table detection
    _, bin_image = cv2.threshold(
        gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # Inverting the image
    bin_image = 255 - bin_image

    # Detect lines (highlighted lines)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_lines = cv2.morphologyEx(
        bin_image, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
    )
    # Hough transformacija za detekciju horizontalnih linija
    horizontal_hough_lines = cv2.HoughLinesP(
        horizontal_lines,
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=minLineLength,
        maxLineGap=10,
    )

    return horizontal_hough_lines


def find_vertical_lines(img, minLineLength=40):
    # Pretvorba u sivu skalu
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding to prepare for table detection
    _, bin_image = cv2.threshold(
        gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # Inverting the image
    bin_image = 255 - bin_image

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    vertical_lines = cv2.morphologyEx(
        bin_image, cv2.MORPH_OPEN, vertical_kernel, iterations=2
    )

    # Hough transformacija za detekciju vertikalnih linija
    vertical_hough_lines = cv2.HoughLinesP(
        vertical_lines,
        1,
        np.pi / 180,
        threshold=20,
        minLineLength=minLineLength,
        maxLineGap=10,
    )

    return vertical_hough_lines


def find_intersections(horizontal_hough_lines, vertical_hough_lines):
    # Pronalaženje tačaka preseka linija
    intersections = []
    if horizontal_hough_lines is not None and vertical_hough_lines is not None:
        for h_line in horizontal_hough_lines:
            for v_line in vertical_hough_lines:
                x1, y1, x2, y2 = h_line[0]
                x3, y3, x4, y4 = v_line[0]
                denominator = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
                if denominator != 0:
                    intersection_x = int(
                        (
                            (x1 * y2 - y1 * x2) * (x3 - x4)
                            - (x1 - x2) * (x3 * y4 - y3 * x4)
                        )
                        / denominator
                    )
                    intersection_y = int(
                        (
                            (x1 * y2 - y1 * x2) * (y3 - y4)
                            - (y1 - y2) * (x3 * y4 - y3 * x4)
                        )
                        / denominator
                    )
                    intersections.append((intersection_x, intersection_y))

    # Sortiranje preseka po x i y koordinatama
    intersections.sort(key=lambda x: (x[0], x[1]))

    return intersections


def cut_table_cells(image):
    height, width, q = image.shape

    horizonal = find_horizonal_lines(image)
    vertical = find_vertical_lines(image)
    intersections = find_intersections(horizonal, vertical)

    horizonal_lines = clean_lines([y for x, y in intersections])

    for row_index in range(len(horizonal_lines) - 1):
        y1, y2 = horizonal_lines[row_index], horizonal_lines[row_index + 1]
        row_image = image[y1:y2, 0:width]
        v_lines = find_vertical_lines(row_image)

        inter = find_intersections(horizonal, v_lines)

        vertical_lines = clean_lines([x for x, y in inter])
        for colume_index in range(len(vertical_lines) - 1):
            x1, x2 = vertical_lines[colume_index], vertical_lines[colume_index + 1]
            height, width, q = row_image.shape
            colume_image = row_image[0:height, x1:x2]

            image_name = f"cell_{row_index}_{colume_index}.jpg"
            cv2.imwrite(f"{output_folder}/{image_name}", colume_image)

    log("Cut table on cells")


def clean_lines(array, cleaning=60):
    resault = [0]
    unique_array = np.unique(array)

    for i in range(len(unique_array)):
        current_nummber = unique_array[i]
        max_nummber = max(resault)

        if max_nummber + cleaning < current_nummber:
            resault.append(current_nummber)

    return resault


def show_image(img):
    cv2.imshow("Display Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def write_to_file(text):
    with open("output.txt", "w") as f:
        f.write(text)

    log("Text wirte in output file.")


# Define a function to extract the numeric part of the filename
def extract_key(filename):
    if filename.startswith("cell_"):
        match = re.search(r"cell_(\d+)_(\d+)\.jpg", filename)
        if match:
            number1 = int(match.group(1))
            number2 = int(match.group(2))
            return (0, number1, number2)
    elif filename.startswith("page_"):
        match = re.search(r"page_(\d+)\.jpg", filename)
        if match:
            number1 = int(match.group(1))
            return (1, number1)
    # For other cases or if the filename format doesn't match, return filename itself
    return (2, filename)


def convert_image_to_text():
    convert_text = ""
    # Get a list of files in the folder
    files = os.listdir(output_folder)

    # Sort filenames using the extracted key
    sorted_filenames = sorted(files, key=extract_key)

    # Loop through all files in the input folder
    for filename in sorted_filenames:
        if filename.endswith((".png", ".jpg", ".jpeg")):
            if filename in skip_page:
                continue
            start_read = time.time()
            # Load the image
            image_path = os.path.join(output_folder, filename)
            image = cv2.imread(image_path)

            # Convert image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Perform OCR using pytesseract
            text = pytesseract.image_to_string(
                gray_image, lang="deu", config=tessdata_dir_config
            )

            convert_text += text + "\n"

            end_read = time.time()
            log(
                f"Read text form image {filename} -> Execute time: {round((end_read - start_read), 2)}s"
            )

    return convert_text


if __name__ == "__main__":
    arguments = sys.argv[1:]  # Ignore the first argument (script name)
    files = os.listdir(input_folder)

    allowed_files = [
        file for file in files if os.path.splitext(file)[1].lower() in allowed_formats
    ]
    for file in allowed_files:
        try:
            clear_folder()
            convert_pdf_to_image(f"{input_folder}/{file}")
            table = find_table_in_image("page_1.jpg")
            cut_table_cells(table)
            textFormImages = convert_image_to_text()

            if save_to_file == True:
                write_to_file(textFormImages)

            os.remove(f"{input_folder}/{file}")
            print(textFormImages)
        except Exception as e:
            print("An error occurred:", e)
