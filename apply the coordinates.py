import pandas as pd
import rasterio
import rasterio.mask
from shapely.geometry import Polygon, box
import os
import numpy as np

# Load the CSV file
file_path = r"D:\\Thesis\\DATA\\Ground Truth\\Turkey 2023 Xiao Yu\\GT_ka_4_bounding_box_coordinates.csv"
df = pd.read_csv(file_path)

# Load the Satellite image
Nepal_Tif = r'C:/Users/sabacomputer/Desktop/New20240730/NDTS_square.tif'
sat_image = rasterio.open(Nepal_Tif)

# Create a directory to save the extracted AOIs
output_path = 'C:/Users/sabacomputer/Desktop/New20240730/applied_polygon4'
os.makedirs(output_path, exist_ok=True)

# List to store extracted data
extracted_data = []

# Function to extract and save AOI given a polygon
def extract_and_save_aoi(polygon, image, output_path, index):
    try:
        # Mask the image with the polygon
        out_image, out_transform = rasterio.mask.mask(image, [polygon], crop=True)
        
        # Define metadata for the output file
        out_meta = image.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        # Save the extracted AOI
        aoi_path = os.path.join(output_path, f'aoi_{index}.tif')
        with rasterio.open(aoi_path, 'w', **out_meta) as dst:
            dst.write(out_image)
        
        print(f"Polygon at index {index} successfully extracted.")
        
        # Collecting the pixel values of each band
        bands_data = []
        for i in range(out_image.shape[0]):
            band_data = out_image[i].flatten()
            band_data = band_data[band_data != 0]  # Exclude zero values
            bands_data.append(band_data)
        
        # Transpose to get the values per pixel across all bands
        pixel_values = np.array(bands_data).T
        all_values = pixel_values.flatten()
        extracted_data.append([index] + list(all_values))
    except ValueError as e:
        print(f"Polygon at index {index} does not intersect with the image bounds: {e}")

# Convert the bounds of the satellite image to a Shapely geometry
image_bounds = box(*sat_image.bounds)

# Check the columns of the DataFrame
print("Columns in CSV:", df.columns)

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Print the current row for debugging
    print(f"Processing row {index}:")
    print(row)
    
    # Extract polygon coordinates from the row
    polygon_coords = []
    for i in range(1, 5):  # Adjusted to handle up to 14 coordinates
        lon_col = f'lon{i}'
        lat_col = f'lat{i}'
        if lon_col in row and lat_col in row:
            try:
                lon = float(row[lon_col])
                lat = float(row[lat_col])
                print(f"Coordinate {i}: lon={lon}, lat={lat}")  # Print each coordinate
                polygon_coords.append((lon, lat))
            except ValueError:
                print(f"Invalid coordinate value at index {index}: lon={row[lon_col]}, lat={row[lat_col]}")
    
    # Print the number of coordinates collected
    print(f"Number of coordinates for polygon at index {index}: {len(polygon_coords)}")

    if len(polygon_coords) < 3:
        print(f"Polygon at index {index} has less than 3 valid coordinates and cannot form a valid polygon.")
        continue

    # Print the coordinates of the polygon
    print(f"Polygon coordinates at index {index}: {polygon_coords}")

    # Create a polygon
    try:
        polygon = Polygon(polygon_coords)
    except Exception as e:
        print(f"Error creating polygon at index {index} with coordinates {polygon_coords}: {e}")
        continue

    # Check if the polygon intersects with the image bounds
    if polygon.intersects(image_bounds):
        # Extract and save the AOI for this building
        extract_and_save_aoi(polygon, sat_image, output_path, index)
    else:
        print(f"Polygon at index {index} is outside the image bounds.")
        print(f"Polygon coordinates: {polygon_coords}")
        print(f"Image bounds: {image_bounds.bounds}")

# Save the extracted data to a CSV file
max_pixel_count = max(len(data) - 1 for data in extracted_data)
column_names = ['Index'] + [f'Value_{i+1}' for i in range(max_pixel_count)]
extracted_data_df = pd.DataFrame(extracted_data, columns=column_names)
output_csv_path = os.path.join(output_path, 'extracted_aoi_pixel_values.csv')
extracted_data_df.to_csv(output_csv_path, index=False)

print("Extraction complete. AOIs saved in:", output_path)
print("Extracted data saved in:", output_csv_path)
