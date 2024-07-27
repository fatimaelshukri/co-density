# co-density
able of contents 

## 1.  Data collection
This section is code to obtain TIF Files from Google Earth Engine (GEE). 

1.1 to obtain the tif imgaes through looping. 
In this program the code uses an API from GEE to open the data set and obtain an image with the given start and end data and with the given longitude and latitude.

1.2 to generate CSV File with min/max carbon monoxide density.
the code for this will a CSV File that contains the minimum and maximum carbon monoxide values  from the images extracted from the GEE.

## 2. Average images

Using the mean average formula, the average image is found by calculating the average carbon monoxide density mol/m^2 (i.e. pixel value) at that point over the range (novemeber 22 2018 - Dec 31 2023). There are 3 types of averages:

- Daily: This creates an average based on the day (e.g. Jan-1-2013, Jan-1-2014, Jan-1-2015 etc.)
- Monthly: This average is based on the pixel values for the complete month form each year
- monthly per year: this creates an average for each month (jan-dec) for its corresponding year thus outpting 12 images for each year between 2018-2023.  
 (72 images in total)
- yearly- a singular averaged image for each year, that is a result after averaing the monthly per year images. ( 2018 to 2023= 6 yearly images ) 
- Annual (Overall): This average is computed using the entire years data (i.e. every image, therefore it is also called the overall average)

## 3. thresholding

The purpose of thresholding is to classify each pixel in an image based on its value, typically by converting it into a binary image where pixels above a certain value (threshold) are set to one color (e.g., white), and those below or equal to the threshold are set to another color (e.g., black). This helps in highlighting specific areas of interest in the image. 

### 4. sobel gradient 

The Sobel Gradient is a method of finding the gradient with respect to its adjacent and diagonally connected neighbours. There is a vertical and horizontal operator, and the magnitude of the gradient is calculated. it highlights areas where there are significant changes in pixel values

## 5. Linear Regression Prediction
Using Linear Regression, the Carbon monoxide density for each month is detected for teh year 2024 and 202.
      5.1 Generate Regression Prediction Images
Induvidual pixels are predicted using pixels from the monthly per images (72 images,from 2018 to 2023) to train the regressison model to predict for the year 2024 and then using those 72 images and the newly formed monthly predicted images for 2024 (84 images) to predict for 2025.

## 6. Calculate RMSE
The Root Mean Square Error (RMSE) for each month is calculated against the real images of 2024. 

## 7. extra codes : 
   7.1  generation of graphs using histograms and the GaussianMixture model (GMM) function to and to depict fluctuations in the mean Carbon Monoxide Density(mol/m^2) throughout the 12 months of each year between 2018-2023

   7.2  colormapping images for visual representation
   
   7.3 to generate missing dates from the GEE to show me the missing dates where there were missing dates that the erath engine could not capture images for that day. (unavailability of datset for that day from the date range specified)

1.  **Data Collection**.
```console
 import ee
 import datetime
 import os
 import geemap  # Ensure you have geemap installed: pip install geemap

"Initialize Earth Engine"
 ee.Initialize()

"Define a region of interest (ROI)"
 roi = ee.Geometry.Polygon(
    [[[50.717442, 24.455254],
      [51.693314, 24.455254],
      [50.717442, 26.226590],
      [51.693314, 26.226590]]])

 "Define the time range"
 start_date = '2022-1-1'
 end_date = '2023-12-31'

"Convert start and end dates to datetime objects"
 start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
 end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')

"Specify the output root folder"
 output_root_folder = r"C:\Users\Wajeeha\Desktop\FAFIOUTPUT"

"Iterate over each day in the specified period"
 current_date = start_datetime
 while current_date < end_datetime:
    # Convert the current date to string
    current_date_str = current_date.strftime('%Y-%m-%d')
    
    # Create a folder for the current month and day
    output_folder = os.path.join(output_root_folder, current_date.strftime('%m-%d'))
    os.makedirs(output_folder, exist_ok=True)
    
    # Sentinel-5P CO dataset
    collection = (ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_CO")
                  .filterBounds(roi)
                  .filterDate(ee.Date(current_date_str), ee.Date(current_date_str).advance(1, 'day'))
                  .select('CO_column_number_density'))

    # Get the mean image
    image = collection.median()

    # Get the minimum and maximum values for the specified region
    stats = image.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=roi,
        scale=1000  # Adjust the scale based on your data
    )

    # Debugging: Print the contents of stats
    print(f"Stats for {current_date_str}: {stats.getInfo()}")

    # Extract min and max values from the statistics for the 'CO_column_number_density' band
    min_value = stats.get('CO_column_number_density_min')
    max_value = stats.get('CO_column_number_density_max')

    # Check if min and max values are not None before using them
    if min_value is not None and max_value is not None:
        try:
            min_value_scaled = min_value.getInfo()
            max_value_scaled = max_value.getInfo()
            print(f"Image {image.id()} - Minimum CO density: {min_value_scaled}, Maximum CO density: {max_value_scaled}")

            # Define the output path
            output_path = os.path.join(output_folder, f"{current_date.strftime('%Y-%m-%d')}.tif")

            # Download the image and save it to the local folder
            geemap.ee_export_image(image, filename=output_path, region=roi, scale=1000)
        except Exception as e:
            print(f"Error processing {current_date_str}: {e}")
    else:
        print(f"No data available for {current_date_str}")
    
    # Move to the next day
    current_date += datetime.timedelta(days=1)

```
   
1.2  **Generating csv file with min and max CO density(mol/m^2)**

```console
import ee
import datetime
import pandas as pd
import numpy as np
import os

"# Initialize Earth Engine"
ee.Initialize()

"" Define a region of interest (ROI)""
roi = ee.Geometry.Polygon(
    [[[50.717442, 24.455254],
      [51.693314, 24.455254],
      [50.717442, 26.226590],
      [51.693314, 26.226590]]])

""# Specify the date range""
start_date_str = '2018-11-22'
end_date_str = '2023-12-31'

""# Convert start and end dates to datetime objects""
start_date = datetime.datetime.strptime(start_date_str, '%Y-%m-%d')
end_date = datetime.datetime.strptime(end_date_str, '%Y-%m-%d')

"" Load existing data if available""
output_csv_path = 'co_data_prediction.csv'
if os.path.exists(output_csv_path):
    co_df = pd.read_csv(output_csv_path)
    existing_dates = set(co_df['Date'])
else:
    co_df = pd.DataFrame()
    existing_dates = set()

"Create an empty list to store the results"
co_data = []

"Create a list to track missing dates"
missing_dates = []

"" Function to check if data already exists for a date""
def data_exists(date_str):
    return date_str in existing_dates

""Iterate over each date in the range""
current_date = start_date
while current_date <= end_date:
    # Convert current date to string format
    current_date_str = current_date.strftime('%Y-%m-%d')

    ########################################## Skip dates for which data already exists
    if data_exists(current_date_str):
        current_date += datetime.timedelta(days=1)
        continue

    # Sentinel-5P CO dataset
    collection = (ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_CO")
                  .filterBounds(roi)
                  .filterDate(ee.Date(current_date_str), ee.Date(current_date_str).advance(1, 'day'))
                  .select('CO_column_number_density'))

    # Get the mean image
    image = collection.median()

    # Get the minimum and maximum values for the specified region
    stats = image.reduceRegion(
        reducer=ee.Reducer.minMax(),
        geometry=roi,
        scale=1000  # Adjust the scale based on your data
    )
    
    # Convert the stats dictionary to a Python dictionary
    stats_info = stats.getInfo()
    
    # Check if the statistics dictionary contains the expected keys
    if 'CO_column_number_density_min' in stats_info and 'CO_column_number_density_max' in stats_info:
        # Extract min and max values from the statistics for the 'CO_column_number_density' band
        min_value = stats_info.get('CO_column_number_density_min')
        max_value = stats_info.get('CO_column_number_density_max')
    
        # Replace None with NaN
        min_value = min_value if min_value is not None else np.nan
        max_value = max_value if max_value is not None else np.nan
    
        # Check if min and max values are valid numbers
        if not np.isnan(min_value) and not np.isnan(max_value):
            # Append the CO data to the list
            co_data.append({'Date': current_date_str,
                            'Min CO Density': min_value,
                            'Max CO Density': max_value})
        else:
            # Log this date as missing
            print(f"No data available for {current_date_str}")
            missing_dates.append(current_date_str)
    else:
        # Log this date as missing
        print(f"No statistics available for {current_date_str}")
        missing_dates.append(current_date_str)

    # Move to the next day
    current_date += datetime.timedelta(days=1)

"" Convert the list of dictionaries to a DataFrame""
new_co_df = pd.DataFrame(co_data)

""Append the new data to the existing DataFrame""
if not new_co_df.empty:
    co_df = pd.concat([co_df, new_co_df], ignore_index=True)

"" Save the updated DataFrame to a CSV file""
if not co_df.empty:
    co_df.to_csv(output_csv_path, index=False)
    print(f"CO data saved to {output_csv_path}")
else:
    print("No CO data available.")

""Display the collected CO data""
print(co_df)

""Display missing dates
if missing_dates:
    print("Missing data for the following dates:")
    for date in missing_dates:
        print(date)

```
## 2. Average images

 **2.1 Daily average**

```console
import os
import re
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

def find_tiff_files(directory):
    """Recursively find all TIFF files in the given directory and its subdirectories."""
    tiff_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.tif'):
                tiff_files.append(os.path.join(root, file))
    return tiff_files

def extract_date_from_filename(filename):
    """Extract the month and day from the filename, assuming format 'YYYY-MM-DD'. This ignores any additional text after the date."""
    try:
        match = re.match(r"(\d{4}-\d{2}-\d{2})", filename)
        if match:
            date_str = match.group(1)
            date = datetime.strptime(date_str, '%Y-%m-%d')
            return date.strftime('%m-%d')
        else:
            print(f"Filename {filename} does not match the expected date format.")
            return None
    except Exception as e:
        print(f"Error extracting date from filename {filename}: {e}")
        return None

def plot_average_image(image_files, output_file=None, save_as_tiff=False, dpi=300):
    if not image_files:
        print("No TIFF files to process.")
        return

    # Open the first image to get dimensions
    first_image_path = image_files[0]
    first_image = tifffile.imread(first_image_path)
    
    if first_image.ndim > 2:
        first_image = first_image[..., 0]
    
    height, width = first_image.shape

    pixel_sum = np.zeros((height, width), dtype=np.float64)
    count = np.zeros((height, width), dtype=np.uint64)

    for image_path in image_files:
        try:
            image = tifffile.imread(image_path).astype(np.float64)
            if image.ndim > 2:
                image = image[..., 0]
        except Exception as e:
            print(f"Skipping file {image_path} due to error: {e}")
            continue

        if image.shape != (height, width):
            print(f"Skipping {image_path} due to different dimensions.")
            continue

        pixel_sum += np.nan_to_num(image)
        count += ~np.isnan(image)

    average_pixels = np.divide(pixel_sum, count, out=np.zeros_like(pixel_sum), where=(count != 0))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(average_pixels, cmap='viridis', interpolation='nearest')
    date_label = os.path.basename(os.path.dirname(image_files[0]))
    plt.title(f'Daily Average CO Column Density - {date_label}', loc='center', fontweight='bold')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Pixel Value', fontweight='bold')
    plt.axis('off')

    if output_file:
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=dpi)

    if save_as_tiff:
        tifffile.imwrite(output_file.replace('.png', '.tif'), average_pixels)

    plt.show()
    plt.close()

def process_images(main_directory, output_directory):
    grouped_files = defaultdict(list)

    tiff_files = find_tiff_files(main_directory)

    for tiff_file in tiff_files:
        filename = os.path.basename(tiff_file)
        date_key = extract_date_from_filename(filename)
        if date_key:
            grouped_files[date_key].append(tiff_file)

    for date_key, file_paths in grouped_files.items():
        output_subdir = os.path.join(output_directory, date_key)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        output_file = os.path.join(output_subdir, f'Average_Image_{date_key}.png')
        plot_average_image(file_paths, output_file=output_file, save_as_tiff=True, dpi=300)

# Specify the directories
main_directory = r"C:\Users\Wajeeha\Desktop\OUPUTCOFAFI"
output_directory = r"C:\Users\Wajeeha\Desktop\AverageImages"

# Process the images
process_images(main_directory, output_directory)

```
**2.2 Monthly average**
```console
import os
import numpy as np
import tifffile as tiff

# Define the directory containing the images
directory = r"C:\Users\Wajeeha\Desktop\AverageImages"
output_directory = r"C:\Users\Wajeeha\Desktop\monty average"

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Function to load and average images for a specific month
def average_images_for_month(directory, month_prefix):
    # Initialize an empty list to store images for averaging
    images = []

    # Iterate over subdirectories
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            # Check if the subdirectory matches the month prefix
            if dir_name.startswith(month_prefix):
                subdir_path = os.path.join(root, dir_name)
                # Get all TIFF files in the subdirectory
                tiff_files = [f for f in os.listdir(subdir_path) if f.endswith('.tif')]
                # Load each TIFF file and append to images list
                for tiff_file in tiff_files:
                    tiff_path = os.path.join(subdir_path, tiff_file)
                    image = tiff.imread(tiff_path)
                    images.append(image)

    if len(images) == 0:
        raise ValueError(f"No TIFF files found for month {month_prefix}")

    # Convert the list of images to a NumPy array
    images = np.array(images)

    # Calculate the average image
    average_image = np.nanmean(images, axis=0)

    if average_image.ndim != 2:
        raise ValueError(f"Invalid shape for average image for month {month_prefix}")

    return average_image

# Save average images for each month as TIFF files
for month_number in range(1, 13):
    month_prefix = f"{month_number:02d}-"
    try:
        average_image = average_images_for_month(directory, month_prefix)
        output_file_path = os.path.join(output_directory, f'{month_prefix}_average_image.tif')
        tiff.imwrite(output_file_path, average_image.astype(np.float32))
        print(f"Saved {output_file_path}")
    except Exception as e:
        print(f"Error processing month {month_prefix}: {e}")

```
**2.3 overall yearly average**

```console
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define the directory containing the monthly average images
directory = r"C:\Users\Wajeeha\Desktop\monty average"

# Define the ROI coordinates
roi_coords = np.array([
    [50.717442, 24.455254],
    [51.693314, 24.455254],
    [51.693314, 26.226590],
    [50.717442, 26.226590],
    [50.717442, 24.455254]
])

def load_monthly_average_images(directory):
    """Load all monthly average TIFF images."""
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tif')]
    images = [tiff.imread(file_path) for file_path in file_paths]
    return images

def generate_yearly_average_image(monthly_average_images):
    """Generate the yearly average image by averaging all monthly average images."""
    yearly_average_image = np.nanmean(np.array(monthly_average_images), axis=0)
    return yearly_average_image

def plot_image_with_roi(image, roi_coords, output_file):
    """Plot the image with ROI outline and save it."""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [50.717442, 51.693314, 24.455254, 26.226590]
    ax.imshow(image, cmap='viridis', extent=extent, origin='upper')
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')

    # Plot ROI polygon as a closed loop
    ax.plot(roi_coords[:, 0], roi_coords[:, 1], color='black', linewidth=2)

    plt.colorbar(ax.images[0], ax=ax, orientation='vertical', fraction=0.046, pad=0.04, label='CO column density mol/m^2')
    plt.title('Yearly Average Carbon Monoxide Density with ROI', fontweight='bold')

    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.1, dpi=500)
    plt.show()

if __name__ == "__main__":
    # Load all monthly average images
    monthly_average_images = load_monthly_average_images(directory)
    
    # Generate the yearly average image
    yearly_average_image = generate_yearly_average_image(monthly_average_images)
    
    # Plot the yearly average image with ROI outline and save it
    output_file = r"C:\Users\Wajeeha\Desktop\yearly_average_with_ROI.png"
    plot_image_with_roi(yearly_average_image, roi_coords, output_file)

```
**2.4 monthly per year images (72 images- 12 for each year from 2018-2023)**

```console
import os
import tifffile as tf
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ROI coordinates for plotting
roi_coords = np.array([
    [50.717442, 24.455254],
    [51.693314, 24.455254],
    [51.693314, 26.226590],
    [50.717442, 26.226590],
    [50.717442, 24.455254]
])

def load_tiff_images(directory):
    """Load all TIFF images from the specified directory and organize by year and month."""
    images_by_month_year = {}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.tif') or file.lower().endswith('.tiff'):
                file_path = os.path.join(root, file)
                date_part = os.path.basename(file).split('.')[0]  # Extract date part from filename
                year, month, day = date_part.split('-')
                year_month = f"{year}-{month}"
                
                if year_month not in images_by_month_year:
                    images_by_month_year[year_month] = []
                
                image = tf.imread(file_path)
                images_by_month_year[year_month].append(image)
    
    return images_by_month_year

def calculate_monthly_averages(images_by_month_year):
    """Calculate the monthly average images."""
    monthly_averages = {}
    
    for year_month, images in images_by_month_year.items():
        # Stack images and calculate mean across the stack
        stacked_images = np.stack(images, axis=0)
        average_image = np.mean(stacked_images, axis=0)
        monthly_averages[year_month] = average_image
    
    return monthly_averages

def save_images(monthly_averages, output_directory):
    """Save the averaged images as TIFF and PNG with ROI outline."""
    for year_month, image in sorted(monthly_averages.items()):
        year, month = year_month.split('-')
        year_folder = os.path.join(output_directory, year)
        os.makedirs(year_folder, exist_ok=True)
        
        # Save the TIFF image
        tiff_filename = os.path.join(year_folder, f"{year_month}.tif")
        tf.imwrite(tiff_filename, image)
        
        # Save the PNG image with colormap and ROI outline
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        extent = [50.717442, 51.693314, 24.455254, 26.226590]
        ax.imshow(image, cmap='viridis', extent=extent, origin='upper')
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')

        # Plot ROI polygon as a closed loop
        ax.plot(roi_coords[:, 0], roi_coords[:, 1], color='black', linewidth=2)
        ax.plot([roi_coords[-1, 0], roi_coords[0, 0]], [roi_coords[-1, 1], roi_coords[0, 1]], color='black', linewidth=2)

        # Add colorbar and title
        cbar = plt.colorbar(ax.images[0], ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('CO column density mol/m^2')
        plt.title(f'Average CO Density for {year_month}', fontweight='bold')

        # Save the PNG image
        png_filename = os.path.join(year_folder, f"{year_month}_ROI.png")
        plt.savefig(png_filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()

# Main directory containing the TIFF images
main_directory = r"C:\Users\Wajeeha\Desktop\OUPUTCOFAFI"
output_directory = r"C:\Users\Wajeeha\Desktop\predictions LR"

# Load TIFF images and organize by year and month
images_by_month_year = load_tiff_images(main_directory)

# Calculate monthly averages
monthly_averages = calculate_monthly_averages(images_by_month_year)

# Save the monthly averaged images as TIFF and PNG
save_images(monthly_averages, output_directory)

```
**2.5 yearly per year from 2018-2023 (6 images for each year)**

```console
import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def load_yearly_average_images(base_directory):
    yearly_average_images = {}

    # Iterate over each year folder
    for year_folder in os.listdir(base_directory):
        year_path = os.path.join(base_directory, year_folder)
        if os.path.isdir(year_path):
            # Initialize list to store monthly images
            monthly_images = []

            # Iterate over each month TIFF file
            for month_file in sorted(os.listdir(year_path)):
                if month_file.endswith('.tif') or month_file.endswith('.tiff'):
                    image_path = os.path.join(year_path, month_file)
                    image = tf.imread(image_path)
                    monthly_images.append(image)

            if monthly_images:
                # Calculate the yearly average image
                yearly_average_image = np.mean(np.stack(monthly_images), axis=0)
                yearly_average_images[year_folder] = yearly_average_image

    return yearly_average_images

def display_yearly_average_images(yearly_average_images, roi_coords):
    plt.figure(figsize=(15, 10))

    for idx, (year, image) in enumerate(sorted(yearly_average_images.items())):
        ax = plt.subplot(2, 3, idx + 1, projection=ccrs.PlateCarree())
        extent = [50.717442, 51.693314, 24.455254, 26.226590]
        im = ax.imshow(image, cmap='viridis', extent=extent, origin='upper')
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')

        # Plot ROI polygon as a closed loop
        ax.plot(roi_coords[:, 0], roi_coords[:, 1], color='white', linewidth=2)
        ax.plot([roi_coords[-1, 0], roi_coords[0, 0]], [roi_coords[-1, 1], roi_coords[0, 1]], color='white', linewidth=2)

        plt.title(f'Yearly Average {year}', fontweight='bold')

    plt.tight_layout()
    plt.show()

# Example usage: Specify the base directory containing yearly folders with monthly average TIFF images
base_directory = r"C:\Users\Wajeeha\Desktop\predictions LR"

# Load yearly average images from monthly averaged images
yearly_average_images = load_yearly_average_images(base_directory)

# Define the coordinates for the ROI (Region of Interest)
roi_coords = np.array([
    [50.717442, 24.455254],
    [51.693314, 24.455254],
    [51.693314, 26.226590],
    [50.717442, 26.226590],
    [50.717442, 24.455254]
])

# Display yearly average images using matplotlib with Viridis colormap and ROI outline
display_yearly_average_images(yearly_average_images, roi_coords)
```

#### 3. Sobel gradient

 **sobel gradient for 6 yearly images**

```console
import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cv2

def load_yearly_average_images(base_directory):
    yearly_average_images = {}

    # Iterate over each year folder
    for year_folder in os.listdir(base_directory):
        year_path = os.path.join(base_directory, year_folder)
        if os.path.isdir(year_path):
            # Initialize list to store monthly images
            monthly_images = []

            # Iterate over each month TIFF file
            for month_file in sorted(os.listdir(year_path)):
                if month_file.endswith('.tif') or month_file.endswith('.tiff'):
                    image_path = os.path.join(year_path, month_file)
                    image = tf.imread(image_path)
                    monthly_images.append(image)

            if monthly_images:
                # Calculate the yearly average image
                yearly_average_image = np.mean(np.stack(monthly_images), axis=0)
                yearly_average_images[year_folder] = yearly_average_image

    return yearly_average_images

def save_tiff_image(image, output_path):
    tf.imwrite(output_path, image)

def save_png_image(image, output_path, roi_coords):
    # Plot the image with Viridis colormap and ROI outline
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [50.717442, 51.693314, 24.455254, 26.226590]
    im = ax.imshow(image, cmap='viridis', extent=extent, origin='upper')
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')

    # Plot ROI polygon as a closed loop
    ax.plot(roi_coords[:, 0], roi_coords[:, 1], color='white', linewidth=2)
    ax.plot([roi_coords[-1, 0], roi_coords[0, 0]], [roi_coords[-1, 1], roi_coords[0, 1]], color='white', linewidth=2)

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('CO column density mol/m^2')
    plt.title('Yearly Average Image', fontweight='bold')

    # Save the plot as PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_sobel_gradient(image, roi_coords):
    # Calculate the gradient using the Sobel operator
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the magnitude of the gradient
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Convert ROI coordinates to image coordinates
    image_shape = image.shape
    roi_coords_image = np.array([
        [(coord[0] - 50.717442) / (51.693314 - 50.717442) * image_shape[1],
         (coord[1] - 24.455254) / (26.226590 - 24.455254) * image_shape[0]]
        for coord in roi_coords
    ], dtype=np.int32)

    # Create ROI mask
    mask = np.zeros_like(gradient_magnitude, dtype=np.uint8)
    cv2.fillPoly(mask, [roi_coords_image], 255)

    # Apply mask to gradient magnitude
    gradient_magnitude_roi = cv2.bitwise_and(gradient_magnitude, gradient_magnitude, mask=mask)

    return gradient_magnitude_roi

def save_sobel_gradient_image(gradient_image, output_path, roi_coords):
    # Plot the Sobel gradient image with ROI outline
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    extent = [50.717442, 51.693314, 24.455254, 26.226590]
    im = ax.imshow(gradient_image, cmap='viridis', extent=extent, origin='upper')
    ax.coastlines(resolution='10m')
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')

    # Plot ROI polygon as a closed loop
    ax.plot(roi_coords[:, 0], roi_coords[:, 1], color='white', linewidth=2)
    ax.plot([roi_coords[-1, 0], roi_coords[0, 0]], [roi_coords[-1, 1], roi_coords[0, 1]], color='white', linewidth=2)

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Gradient Magnitude', fontweight='bold')
    plt.title('Sobel Gradient Image', fontweight='bold')

    # Save the plot as PNG
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_yearly_average_images(yearly_average_images, base_directory, roi_coords):
    for year_folder, image in yearly_average_images.items():
        # Save yearly average image as TIFF
        tiff_output_path = os.path.join(base_directory, year_folder, f'{year_folder}_yearly_average.tif')
        save_tiff_image(image, tiff_output_path)

        # Save PNG version with Viridis colormap and ROI outline
        png_output_path = os.path.join(base_directory, year_folder, f'{year_folder}_yearly_average_with_ROI.png')
        save_png_image(image, png_output_path, roi_coords)

        # Calculate and save Sobel gradient with ROI outline
        gradient_image = calculate_sobel_gradient(image, roi_coords)
        sobel_output_path = os.path.join(base_directory, year_folder, f'{year_folder}_sobel_gradient_with_ROI.png')
        save_sobel_gradient_image(gradient_image, sobel_output_path, roi_coords)

# Example usage: Specify the base directory containing yearly folders with monthly average TIFF images
base_directory = r"C:\Users\Wajeeha\Desktop\predictions LR"

# Load yearly average images from monthly averaged images
yearly_average_images = load_yearly_average_images(base_directory)

# Define the coordinates for the ROI (Region of Interest)
roi_coords = np.array([
    [50.717442, 24.455254],
    [51.693314, 24.455254],
    [51.693314, 26.226590],
    [50.717442, 26.226590],
    [50.717442, 24.455254]
])

# Process and save yearly average images, PNG versions, and Sobel gradients with ROI outline
process_yearly_average_images(yearly_average_images, base_directory, roi_coords)
```
## 4. Thresholding
```console
import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def process_and_display_thresholded_image(image_path, threshold):
    # Load the TIFF image
    image_data = tf.imread(image_path)
    
    # Create subplots for the original and thresholded images
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))

    # Original image plot
    axs[0].imshow(image_data, cmap='viridis')
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Initialize an empty binary image (3D array for RGB representation)
    binary_image = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.uint8)
    white_pixels = 0
    nan_pixels = 0

    # Iterate through each pixel and apply the threshold condition
    for row in range(image_data.shape[0]):
        for col in range(image_data.shape[1]):
            value = image_data[row, col]

            if np.isnan(value):
                # Set NaN values to red (255, 0, 0)
                binary_image[row, col] = [255, 0, 0]
                nan_pixels += 1  # Increment NaN pixel count
            elif value > threshold:
                # Set values above threshold to white (255, 255, 255)
                binary_image[row, col] = [255, 255, 255]
                white_pixels += 1  # Increment white pixel count
            else:
                # Set values below or equal to threshold to black (0, 0, 0)
                binary_image[row, col] = [0, 0, 0]

    # Calculate total valid pixels (excluding NaN)
    total_valid_pixels = image_data.size - nan_pixels

    # Calculate white pixel ratio (white pixels / total valid pixels)
    white_pixel_ratio = (white_pixels / total_valid_pixels) * 100 if total_valid_pixels > 0 else 0

    # Plot the thresholded image
    axs[1].imshow(binary_image)
    axs[1].set_title(f'Threshold = {threshold}\nWhite Pixel Ratio: {white_pixel_ratio:.2f}%')
    axs[1].axis('off')

    # Display the plot
    plt.show()

def process_images_in_directory(main_directory, threshold):
    for year_folder in os.listdir(main_directory):
        year_path = os.path.join(main_directory, year_folder)
        if os.path.isdir(year_path):
            for file_name in os.listdir(year_path):
                if file_name.endswith('.tif'):
                    image_path = os.path.join(year_path, file_name)
                    process_and_display_thresholded_image(image_path, threshold)

# Define the main directory containing the subfolders
main_directory = r"C:\Users\Wajeeha\Desktop\predictions LR"

# Define a single threshold value
threshold = 0.0357

# Process and display all images in the directory with the specified threshold
process_images_in_directory(main_directory, threshold)
```


### 5. Linear regression

```console
import os
import numpy as np
import tifffile as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def load_tiff_images(directory, year_range):
    """Load TIFF images for the specified year range."""
    images = []
    for year in year_range:
        year_dir = os.path.join(directory, str(year))
        for month in range(1, 13):
            file_path = os.path.join(year_dir, f"{year}-{month:02d}.tif")
            if os.path.exists(file_path):
                image = tf.imread(file_path)
                images.append((year, month, image))
            else:
                print(f"File not found: {file_path}")
                images.append((year, month, None))
    return images

def retrieve_pixel_values(images, month, x, y):
    """Retrieve pixel values from the images at position (x, y) for a specific month."""
    pixel_values = []
    for year, image_month, image in images:
        if image_month == month and image is not None and 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
            pixel_value = image[x, y]
            if np.isscalar(pixel_value) and not np.isnan(pixel_value):
                pixel_values.append(pixel_value)
    return pixel_values

def compute_linear_regression(pixel_values):
    """Predict the next pixel value using linear regression."""
    if not pixel_values:
        return np.nan

    X = np.arange(len(pixel_values)).reshape(-1, 1)  # Reshape for scikit-learn
    y = np.array(pixel_values)

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict the value following the last data point
    next_index = len(pixel_values)
    next_value = model.predict([[next_index]])[0]

    return next_value

def predict_images_for_year(images, year, output_directory):
    """Predict images for a specified year using the historical data."""
    predictions = {}
    all_actual_values = []
    all_predicted_values = []

    for month in range(1, 13):
        monthly_images = valid_images_by_month[month]
        if monthly_images:
            height, width = monthly_images[0].shape
            predicted_image = np.empty((height, width))

            for x in range(height):  # Iterate over rows
                for y in range(width):  # Iterate over columns
                    pixel_values = retrieve_pixel_values(images, month, x, y)
                    next_value = compute_linear_regression(pixel_values)
                    predicted_image[x, y] = next_value

                    # Collect the actual and predicted values
                    if len(pixel_values) > 0:
                        all_actual_values.append(pixel_values[-1])  # Actual value (last known value)
                        all_predicted_values.append(next_value)  # Predicted value

            # Store predicted image
            predictions[month] = predicted_image

            # Save the predicted image
            output_path = os.path.join(output_directory, f"predicted_{year}_{month:02d}.tif")
            tf.imwrite(output_path, predicted_image)

    return predictions, all_actual_values, all_predicted_values

# Directories
main_directory = r"C:\Users\Wajeeha\Desktop\predictions LR"
predicted_directory = r"C:\Users\Wajeeha\Desktop\2k24+25 revised"

# Load all historical images (2018-2023)
year_range = range(2018, 2024)
images = load_tiff_images(main_directory, year_range)

# Initialize dictionary to group images by month
valid_images_by_month = {month: [] for month in range(1, 13)}

# Group images by month
for year, month, image in images:
    if image is not None:
        valid_images_by_month[month].append(image)

# Predict images for 2024
predictions_2024, actual_values_2024, predicted_values_2024 = predict_images_for_year(images, 2024, predicted_directory)

# Update images list to include 2024 predictions
for month, predicted_image in predictions_2024.items():
    images.append((2024, month, predicted_image))

# Predict images for 2025 using updated data
predictions_2025, actual_values_2025, predicted_values_2025 = predict_images_for_year(images, 2025, predicted_directory)

# Combine actual and predicted values for error metrics
all_actual_values = actual_values_2024 + actual_values_2025
all_predicted_values = predicted_values_2024 + predicted_values_2025

# Ensure that actual and predicted values are not empty
if all_actual_values and all_predicted_values:
    # Calculate MSE, MAE, R², and RMSE
    mse = mean_squared_error(all_actual_values, all_predicted_values)
    mae = mean_absolute_error(all_actual_values, all_predicted_values)
    r2 = r2_score(all_actual_values, all_predicted_values)
    rmse = np.sqrt(mse)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R²): {r2}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Fit a linear regression model for plotting the regression line
    model = LinearRegression()
    X = np.array(all_actual_values).reshape(-1, 1)
    y = np.array(all_predicted_values)
    model.fit(X, y)
    regression_line = model.predict(X)

    # Plot the actual values vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(all_actual_values, all_predicted_values, color='blue', label='Predicted vs Actual')
    plt.plot(all_actual_values, regression_line, color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Actual CO Density')
    plt.ylabel('Predicted CO Density')
    plt.title('Regression Line vs Predicted Values')
    plt.legend()
    plt.show()

else:
    print("Error: Actual or predicted values are empty.")

```
  **5.1 colormapping the predicted images**

```console
import os
import tifffile as tiff
import matplotlib.pyplot as plt

def save_colormapped_image(image_path, output_directory, colormap='viridis'):
    """Save a colormapped TIFF image with only the intensity colorbar on the right."""
    # Load the TIFF file
    image = tiff.imread(image_path)
    
    # Extract year and month from the file name
    file_name = os.path.basename(image_path)
    year_month = file_name.split('_')[1:3]
    year = year_month[0]
    month = year_month[1].split('.')[0]  # Remove the '.tif' extension
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Display the image with colormap
    img = ax.imshow(image, cmap=colormap, origin='upper')
    
    # Add colorbar on the right
    cbar = plt.colorbar(img, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Intensity')
    
    # Ensure only the colorbar is visible
    cbar.ax.yaxis.set_ticks_position('none')  # Remove ticks on the colorbar
    cbar.ax.yaxis.set_tick_params(width=0)    # Ensure no tick marks

    # Set the title
    plt.title(f'Predicted Image of {year}-{month}')
    
    # Save the image to the specified output directory
    output_path = os.path.join(output_directory, file_name)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the figure to free memory

def process_and_save_images(input_directory, output_directory, colormap='viridis'):
    """Process and save all TIFF images in the input directory to the output directory."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)  # Create the output directory if it doesn't exist
    
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.tif'):
            image_path = os.path.join(input_directory, file_name)
            print(f"Saving {image_path} to {output_directory}")
            save_colormapped_image(image_path, output_directory, colormap)

# Define your directories
input_directory = r"C:\Users\Wajeeha\Desktop\2k24+25 revised"
output_directory = r"C:\Users\Wajeeha\Desktop\Colormapped Images"

# Process and save all images
process_and_save_images(input_directory, output_directory, colormap='viridis')

```
## 6. regression validation 
**6.1 to calculate the Root mean squraed error (RMSE) against the real images of 2024**

```console
import os
import numpy as np
import tifffile as tf
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd

def load_predicted_images(directory, months):
    """Load predicted TIFF images for the specified months."""
    images = []
    for month in months:
        file_path = os.path.join(directory, f"predicted_{month:02d}-2024.tif")
        if os.path.exists(file_path):
            image = tf.imread(file_path)
            images.append(image)
        else:
            images.append(None)
    return images

def load_real_images(directory, months):
    """Load real TIFF images for the specified months."""
    images = []
    for month in months:
        file_path = os.path.join(directory, f"{month:02d}_average_image.tif")
        if os.path.exists(file_path):
            image = tf.imread(file_path)
            images.append(image)
        else:
            images.append(None)
    return images

def compute_rmse(predicted_images, real_images):
    """Compute RMSE for each pair of predicted and real images."""
    rmse_values = []
    for pred_img, real_img in zip(predicted_images, real_images):
        if pred_img is not None and real_img is not None:
            rmse = sqrt(mean_squared_error(real_img.flatten(), pred_img.flatten()))
            rmse_values.append(rmse)
    return rmse_values

def process_images(main_directory, output_directory, months):
    # Predicted images for January to June 2024
    predicted_images = load_predicted_images(main_directory, months)

    # Real images for January to June
    real_images = load_real_images(output_directory, months)

    # Compute RMSE values
    rmse_values = compute_rmse(predicted_images, real_images)

    # Create a DataFrame for the results
    results = pd.DataFrame({
        'Month': months,
        'RMSE': rmse_values
    })

    # Print results
    print("RMSE for each month:")
    print(results)

    if len(rmse_values) > 0:
        # Calculate mean RMSE and standard deviation
        mean_rmse = np.mean(rmse_values)
        std_rmse = np.std(rmse_values)
        print("\nMean RMSE:", mean_rmse)
        print("Standard Deviation RMSE:", std_rmse)
    else:
        print("\nNo valid RMSE values calculated.")

    return results, mean_rmse, std_rmse

# Directories
main_directory = r"C:\Users\Wajeeha\Desktop\predictions"
output_directory = r"C:\Users\Wajeeha\Desktop\2k24"  # Directory for real images
s
# Months to process (January to June)
months = [1, 2, 3, 4, 5, 6]

#Process images and compute RMSE
results, mean_rmse, std_rmse = process_images(main_directory, output_directory, months)
```

## 7. Extra Codes
*7.1 graphical representation of deviations in CO density using histogram and GMM function**

```console
import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def load_monthly_images(main_directory, start_year, end_year):
    """Load monthly TIFF images for the specified year range."""
    images = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == 2018 and month < 11:
                continue  # Skip months before November 2018
            file_path = os.path.join(main_directory, f"{year}", f"{year}-{month:02d}.tif")
            if os.path.exists(file_path):
                image = tf.imread(file_path)
                images.append((year, month, image))
    return images

def fit_gmm(pixel_values, n_components=2):
    """Fit a Gaussian Mixture Model to the pixel values."""
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(pixel_values.reshape(-1, 1))
    return gmm

def generate_combined_histogram(images):
    """Generate a combined histogram for the pixel values of monthly images with GMM fit."""
    colors = plt.cm.get_cmap('tab20', 12)  # Get a colormap with 12 distinct colors
    combined_pixel_values = {i: [] for i in range(1, 13)}
    
    for year, month, image in images:
        if image is not None:
            combined_pixel_values[month].extend(image.flatten())
    
    plt.figure(figsize=(14, 8))
    
    for month in range(1, 13):
        pixel_values = np.array(combined_pixel_values[month])
        plt.hist(pixel_values, bins=50, color=colors(month - 1), alpha=0.6, label=f'Month {month}', density=True)
        
        if len(pixel_values) > 0:
            gmm = fit_gmm(pixel_values)
            x = np.linspace(pixel_values.min(), pixel_values.max(), 1000).reshape(-1, 1)
            logprob = gmm.score_samples(x)
            pdf = np.exp(logprob)
            plt.plot(x, pdf, color=colors(month - 1), linewidth=2)
    
    plt.title('Combined Pixel Value Histogram for Months from Nov 2018 to Dec 2023 with GMM Fit')
    plt.xlabel('Pixel Value')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

# Main directory
main_directory = r"C:\Users\Wajeeha\Desktop\predictions LR"

# Year range
start_year = 2018
end_year = 2023

# Load monthly TIFF images
images = load_monthly_images(main_directory, start_year, end_year)

# Generate combined histogram
generate_combined_histogram(images)
```

**7.2 generating missing dates**

```console
import ee
import datetime

# Initialize Earth Engine
ee.Initialize()

# Define a region of interest (ROI)
roi = ee.Geometry.Polygon(
    [[[50.717442, 24.455254],
      [51.693314, 24.455254],
      [50.717442, 26.226590],
      [51.693314, 26.226590]]])

def check_dataset_availability(start_date, end_date):
    # Convert start and end dates to datetime objects
    start_datetime = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.datetime.strptime(end_date, '%Y-%m-%d')

    # Create a list to store missing dates
    missing_dates = []

    # Iterate over each day in the specified period
    current_date = start_datetime
    while current_date < end_datetime:
        # Convert the current date to string
        current_date_str = current_date.strftime('%Y-%m-%d')

        # Sentinel-5P CO dataset
        collection = (ee.ImageCollection("COPERNICUS/S5P/NRTI/L3_CO")
                      .filterBounds(roi)
                      .filterDate(ee.Date(current_date_str), ee.Date(current_date_str).advance(1, 'day'))
                      .select('CO_column_number_density'))

        # Check if collection is empty
        if collection.size().getInfo() == 0:
            missing_dates.append(current_date_str)

        # Move to the next day
        current_date += datetime.timedelta(days=1)

    return missing_dates
# Define the time range
start_date = '2018-11-22'
end_date = '2023-12-31'
# Check dataset availability
missing_dates = check_dataset_availability(start_date, end_date)
# Print missing dates
if missing_dates:
    print("Missing dates:")
    for date in missing_dates:
        print(date)
else:
    print("No missing dates.")
```
