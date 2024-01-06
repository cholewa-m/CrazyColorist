import cv2
import sys
import os
import numpy as np
import random


LOWER_S = 11
UPPER_S = 255
LOWER_V = 11
UPPER_V = 244


# noinspection PyTypeChecker
def fill_contours_mask(contour_mask, contours):
    cv2.drawContours(
        image=contour_mask,
        contours=contours,
        contourIdx=-1,  # all
        color=255,
        thickness=-1  # fill
    )


def create_contour_mask(image, contour):
    contour_mask = np.zeros_like(image[:, :, 0])
    fill_contours_mask(contour_mask, [contour])
    return contour_mask


def color_contour(image, contour, color):
    contour_mask = create_contour_mask(image, contour)
    image[:, :, 0][contour_mask == 255] = color


def random_exchange_colors(image, contours_and_colors):
    random.shuffle(contours_and_colors)
    for i in range(len(contours_and_colors) - 1):
        curr_contour, curr_color = contours_and_colors[i]
        next_contour, next_color = contours_and_colors[i + 1]
        color_contour(image, curr_contour, next_color)
        color_contour(image, next_contour, curr_color)


def avg_hue_for_contour(image, contour):
    contour_mask = create_contour_mask(image, contour)
    mean_hue = np.mean(image[:, :, 0][contour_mask == 255])
    return mean_hue


def get_color_range_contours(image, lower_h, upper_h, lower_s, upper_s, lower_v, upper_v):
    lower = np.array([lower_h, lower_s, lower_v])
    upper = np.array([upper_h, upper_s, upper_v])
    mask = cv2.inRange(image, lower, upper)
    contours, _ = cv2.findContours(  # contours as a list of points
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE  # get external contours, reduce complexity (less points)
    )
    return contours


def get_all_contours_and_colors(image, color_ranges):
    contours_and_colors = []
    for (lower_h, upper_h) in color_ranges:
        contours = get_color_range_contours(image, lower_h, upper_h, LOWER_S, UPPER_S, LOWER_V, UPPER_V)
        for contour in contours:
            mean_hue = avg_hue_for_contour(image, contour)
            contours_and_colors.append((contour, mean_hue))
    print(f" ->Found {len(contours_and_colors)} contours")
    return contours_and_colors


def get_color_ranges(color_ranges_number):
    color_ranges = []
    progress = int(180 / color_ranges_number)
    for i in range(0, 180, progress):
        color_ranges.append((i, i + progress))
    return color_ranges


def colorize(image, color_ranges_number):
    color_ranges = get_color_ranges(color_ranges_number)
    contours_and_colors = get_all_contours_and_colors(image, color_ranges)
    random_exchange_colors(image, contours_and_colors)
    return image


def crazy_colorist(path, color_ranges_number):
    image = cv2.imread(path)
    if image is None:
        print("Couldn't load the image!")
        return
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    colorized_image = colorize(hsv_image, color_ranges_number)
    output_image = cv2.cvtColor(colorized_image, cv2.COLOR_HSV2BGR)
    base_name = os.path.basename(path)
    output_image_path = "CrazyColorized_" + base_name
    cv2.imwrite(output_image_path, output_image)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error!\n "
              "->Use: python3 crazy_colorist.py [path_to_image] [number_of_color_ranges]\n"
              "->Supported formats: jpg, png, bmp")
    else:
        input_path = sys.argv[1]
        try:
            num_color_ranges = int(sys.argv[2])
        except (ValueError, IndexError):
            num_color_ranges = 15
        crazy_colorist(input_path, num_color_ranges)
