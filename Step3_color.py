#hkb
#__author__="harpreet kaur bargota"
#__email__="harpreet.bargota@agr.gc.ca"
#__Project__="Faba bean Feature extraction pipeline (Step3)"
#References:
#https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2lab
#https://matplotlib.org/stable/gallery/color/named_colors.html#css-colors
#https://docs.scipy.org/doc/scipy/reference/spatial.distance.html

"""This Step3 processes images from folder based on bounding box (bbox-0,1,2,3) coordinates stored in a .csv file of dimensional & shape features from output directory of step2.
It converts the normalized RGB value to the closest named color from the CSS4 color set using the CIE Lab color space for better perceptual accuracy.
 and counts the RGB color name. The maximum count of color name (excluding the blue background) is the dominant color and the dominant RGB value. Finally
  new columns for dominant RGB value and dominant color of the bean is cretaed in the in the dimensional and shape features dataframe, which is 
  saved to a .csv file in the output directory."""

    


import argparse
import os
from collections import Counter
import numpy as np
import pandas as pd
import cv2
from skimage.color import rgb2lab
from matplotlib.colors import CSS4_COLORS
from scipy.spatial.distance import cdist

# Function to map an RGB color to a CSS4 color name
def rgb_to_css4_color_name(rgb):
    
    """ Converts the normalized RGB value to the closest named color from the CSS4 color 
        set using the CIE Lab color space for better perceptual accuracy. 
            
        Arguments: RGB value """

    rgb_normalized = np.array(rgb) / 255  # Normalize the scale to 0-1
    lab_color = rgb2lab(np.array([[rgb_normalized]]))[0][0]
    css_colors_lab = np.array([
        rgb2lab(np.array([[np.array([
            int(c[1:3], 16), int(c[3:5], 16), int(c[5:], 16)]) / 255.0]]))[0][0]
        for c in CSS4_COLORS.values()
    ])
    
   
    colors_distances = cdist([lab_color], css_colors_lab)  # Compute distances
    closest_color_index = np.argmin(colors_distances)  # Find the closest lab color
    return list(CSS4_COLORS.keys())[closest_color_index]  # Get the color name

# Function to find the most frequent RGB tuple in an ROI, excluding blue background
def get_dominant_color_excluding_blue(image, x, y, w, h):

    """Extracts the most dominant color i.e. the maximum count of color name from bounding box coordinates of bean boxes,
         while excluding blue background color (specifically "cornflowerblue" and "dodgerblue").
         
         Argumnrts: image and bounding box coordinates x,y,w,h"""
    roi = image[int(y):int(y + h), int(x):int(x + w)]  # Crop the region
    roi_reshaped = roi.reshape(-1, 3)  # Flatten the ROI to RGB tuples
    color_counts = Counter(map(tuple, roi_reshaped)).most_common()

    # Iterate through the most common colors to exclude blue background
    for color, _ in color_counts:
        color_name = rgb_to_css4_color_name(color)
        if color_name.lower() not in ["cornflowerblue", "dodgerblue"]:
            return color  # Return the first non-blue color

    return color_counts[0][0]  # If all are blue, return the most common




def process_color(image_folder, output_folder):

    """ Processes images from folder based on bounding box (bbox) coordinates stored in a .csv file of dimensional & shape features.
          It extracts the dominant RGB value and the color of the specified regions in each image while excluding blue background shades and saves the results in new column
          in the dataframe which is saved to a .csv file in the output directory.
          
          Arguments: Input directory of images, output directory of step2 which contains the .csv file of dimensional and shape features

          Raises:
            TypeError: If the conditions are not met.

          """
    # Read the .csv file from the output folder obtained from the previous step in the pipeline
    csv_files = [f for f in os.listdir(output_folder) if f.endswith('.csv')]

    # Raise an error if there is no .csv file
    if not csv_files:
        raise FileNotFoundError(f"No .csv file found in folder: {output_folder}")

    # Specify the path to the .csv file, assuming that only one .csv file is there
    csv_path = os.path.join(output_folder, csv_files[0])

    # Read .csv file into DataFrame
    df_image = pd.read_csv(csv_path)

    # Create empty lists for color names and RGB values
    color_names = []
    RGB_values = []

    # Iterate through the dataframe for each value of bbox coordinates                         
    for index, row in df_image.iterrows():
        x = row["bbox-1"]  # x-coordinate
        y = row["bbox-0"]  # y-coordinate
        w = row["bbox-3"] - row["bbox-1"]  # Width of the image
        h = row["bbox-2"] - row["bbox-0"]  # Height of the image

        # For each value of bbox, read the image from the image path according to the class name as specified in the dataframe
        image_path = os.path.join(image_folder, f"{row['Class']}.JPG")  # Define path according to class name for each bbox value
                
        image = cv2.imread(image_path)  # Read the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        
        dominant_color = get_dominant_color_excluding_blue(image, x, y, w, h)  # Exclude blue background

        RGB_values.append(dominant_color)  # Get the RGB value and add it to the RGB list

        color_name = rgb_to_css4_color_name(dominant_color)  # Convert RGB to CSS4 color name
        color_names.append(color_name)  # Add the color names to a list

    # Create a new column in dataframe with RGB values    
    df_image["RGB value of Seed"] = RGB_values     

    # Create a new column in dataframe with color names  
    df_image["color_seeds"] = color_names

    print (df_image)

    # Save the final file of feature extraction from all images into the output folder
    output_filename = "FE_Color.csv"
    output_path = os.path.join(output_folder, output_filename)
    df_image.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and analyze dataframe with features extracted to get the final .csv file with features and color of beans")
    parser.add_argument("image_folder", help="Path to the images")
    parser.add_argument("output_folder", help="Path to the output folder")

    args = parser.parse_args()
    process_color(args.image_folder, args.output_folder)
