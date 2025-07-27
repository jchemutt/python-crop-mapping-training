# Introduction to Python Programming with Crop Type Mapping Focus

## Python Basics

### Introduction to Python Syntax

Python syntax refers to the set of rules that defines how Python programs are written and interpreted.
It is known for being clear and human-readable. Python uses indentation instead of braces to define code blocks.

```python
# This is a comment
print("Hello, world!")  # Prints a message

if True:
    print("This is indented correctly")

# Incorrect indentation
if True:
print("This will raise an IndentationError")
```

```python
# Simple Control Flow Example
weather = "rainy"

if weather == "rainy":
    print("Crop might be overwatered.")
else:
    print("Irrigation might be needed.")
```

**Running Python Code:**

- Save the script as `.py` and run from terminal: `python my_script.py`
- Or use Jupyter Notebook / VS Code

**Exercise:**  
Create `hello_crop.py` that prints a welcome message and current year using the datetime module.

```python
import datetime
print("Welcome to Crop Type Mapping with Python!")
print("Year:", datetime.datetime.now().year)
```

### Variables, Data Types, and Operations

In Python, **variables** are used to store data that your program can access and manipulate.  
Each variable has a **name** and holds a **value**, which can be of different **data types** such as:

- Integers (e.g., `5`)
- Floating-point numbers (e.g., `3.14`)
- Strings (e.g., `'Maize'`)
- Booleans (e.g., `True`, `False`)
- Collections (e.g., lists, dictionaries, tuples)

These data types are crucial in crop mapping tasks, where you handle numeric values (yield, area), names of crops, and logic checks.

---

#### Declaring Variables

Variables are created by **assigning a value** using the `=` operator.

```python
# Assigning values to variables
crop = 'Maize'           # String
area = 5                 # Integer
yield_per_hectare = 2.8  # Float
is_irrigated = True      # Boolean
```

---

#### Data Types in Practice

Python supports several built-in data types. Here are the most common ones you'll encounter in geospatial or agricultural analysis:

| Type    | Description                    | Example               |
| ------- | ------------------------------ | --------------------- |
| `int`   | Integer numbers                | `5`, `-10`, `2024`    |
| `float` | Decimal numbers                | `2.5`, `0.01`         |
| `str`   | Text or characters             | `'Maize'`, `'Plot A'` |
| `bool`  | Logical true/false             | `True`, `False`       |
| `list`  | Ordered, changeable collection | `['Maize', 'Beans']`  |
| `dict`  | Key-value pairs                | `{'Maize': 2.5}`      |

---

#### Basic Operations

You can perform arithmetic or logical operations with variables.

```python
# Arithmetic
total_yield = area * yield_per_hectare
print("Total Yield (tons):", total_yield)

# String concatenation
crop_info = "Crop: " + crop
print(crop_info)

# Boolean logic
if is_irrigated:
    print("Field is irrigated")
else:
    print("Field is rainfed")
```

---

#### Type Conversion

Sometimes, you need to **convert values** from one type to another using functions like `int()`, `float()`, `str()`, or `bool()`.

```python
value = "25"
value_int = int(value)  # Convert string to integer
print(value_int + 5)
```

---

#### Exercises

Try these on your own or in a Jupyter Notebook:

1. Declare variables for a crop name, area of land (in hectares), and yield per hectare.
2. Compute and print the total expected yield.
3. Convert a number given as a string (e.g., `'10'`) into an integer and multiply it by 2.
4. Write a condition that prints "High Yield" if yield > 3, "Moderate" if between 2 and 3, else "Low".

### Lists, Dictionaries, and Tuples

Python provides several built-in data structures to store collections of items.  
For crop mapping and geospatial tasks, the most commonly used ones are:

- `list`: ordered, changeable collection
- `tuple`: ordered, unchangeable collection
- `dict`: key-value mappings

---

#### Lists

Lists are **ordered** and **mutable** collections that can store any type of data.

```python
# Define a list of crops
crops = ['Maize', 'Beans', 'Sorghum']

# Access items
print(crops[0])  # Output: Maize

# Add new item
crops.append('Cassava')

# Check contents
print(crops)
```

**Common List Methods**

- `append()`: Add item to end
- `remove()`: Remove by value
- `sort()`: Sort items
- `len()`: Get number of items

---

#### Tuples

Tuples are like lists, but **immutable** (they cannot be changed after creation).  
Use them to store **fixed** data like coordinates.

```python
# Define a tuple
coordinates = (1.95, 30.06)

# Access values
print("Latitude:", coordinates[0])
```

---

#### Dictionaries

Dictionaries store data in **key-value** pairs.  
They’re ideal for mapping relationships, such as crop name → yield.

```python
# Create a dictionary of crop yields
crop_yields = {
    'Maize': 2.5,
    'Beans': 1.8
}

# Access value by key
print(crop_yields['Maize'])  # Output: 2.5

# Add new entry
crop_yields['Sorghum'] = 2.0

# Display all keys
print(crop_yields.keys())
```

**Common Dictionary Methods**

- `get(key)`: Get value safely
- `keys()`: Return all keys
- `values()`: Return all values
- `items()`: Return key-value pairs

---

#### Exercises

1. Create a list of five crops and print the third crop.
2. Create a tuple representing the lat/lon of a farm location.
3. Create a dictionary with crop names as keys and yield per hectare as values.
4. Add a new crop to the dictionary and print all items.
5. Write a function that takes a crop name and returns the yield from the dictionary.

### Loops and Conditionals in Python

Loops and conditionals allow you to control the **flow of logic** in your program.  
They are essential when dealing with datasets, performing repetitive tasks, or applying filters in crop analysis.

---

#### Conditional Statements (`if`, `elif`, `else`)

Use conditionals to execute different blocks of code based on logic.

```python
yield_per_hectare = 2.5

if yield_per_hectare > 3:
    print("High yield")
elif yield_per_hectare > 2:
    print("Moderate yield")
else:
    print("Low yield")
```

**Tips**:

- Use `==` for equality, `!=` for inequality
- Use `and`, `or`, `not` for combining conditions

---

#### `for` Loops

Use a `for` loop to iterate over items in a list or a range of numbers.

```python
crops = ['Maize', 'Beans', 'Sorghum']

for crop in crops:
    print("Processing crop:", crop)

# Loop with range
for i in range(3):
    print("Iteration:", i)
```

---

#### `while` Loops

`while` loops run as long as a condition remains true.

```python
counter = 0

while counter < 3:
    print("Count:", counter)
    counter += 1
```

Be careful with infinite loops — always ensure your loop has an exit condition.

---

#### Loop Control: `break` and `continue`

- `break`: exits the loop entirely
- `continue`: skips to the next iteration

```python
for crop in crops:
    if crop == 'Beans':
        continue  # Skip Beans
    print(crop)

# Breaking the loop
for crop in crops:
    if crop == 'Sorghum':
        break
    print(crop)
```

---

#### Exercises

1. Write a condition to classify yield as `"Low"`, `"Medium"`, or `"High"`.
2. Loop through a list of crops and print each in uppercase.
3. Use a `while` loop to print numbers from 1 to 5.
4. Break the loop when the crop `"Sorghum"` is found.
5. Combine conditionals with a loop to only print crops with yield > 2.0 tons/ha.

### Functions and File I/O in Python

Functions and file input/output (I/O) are essential tools in Python that allow you to:

- Reuse blocks of code
- Organize logic into manageable units
- Read and write files for automation and reporting

---

#### Defining and Using Functions

Functions are declared using the `def` keyword.  
They can accept **parameters** and return **results**.

```python
# Define a function to calculate yield
def calculate_yield(area, rate):
    return area * rate

# Use the function
result = calculate_yield(10, 2.5)
print("Total Yield:", result)
```

**Tips**:

- Functions improve modularity and reusability.
- You can return multiple values using tuples.

---

#### Built-in Python Functions

Python provides a rich set of built-in functions:

```python
crops = ['Maize', 'Beans']

print(len(crops))       # Count items
print(type(crops))      # Type of variable
print(sorted(crops))    # Sorted list
```

---

#### Reading from Files

Use the built-in `open()` function to read content from files (e.g., text or CSV).

```python
# Reading a file line by line
with open('crop_names.txt', 'r') as file:
    for line in file:
        print("Crop:", line.strip())
```

**Common modes**:

- `'r'`: read
- `'w'`: write (overwrites)
- `'a'`: append
- `'rb'`: read binary

---

#### Writing to Files

To save data or reports to a file:

```python
# Writing to a file
with open('output.txt', 'w') as file:
    file.write("Crop Yield Report\n")
    file.write("Maize: 2.5 tons/ha\n")
```

Always use `with open(...)` — it automatically closes the file safely.

---

#### Exercises

1. Write a function that calculates the cost of production given area and cost per hectare.
2. Create a function that returns the longer of two crop names.
3. Read a file with crop names and print each with `"Crop:"` prefix.
4. Write a function that takes a list of yields and writes them to a file.
5. Extend the above function to also return the average yield.

## Working with Geospatial Data

### Understanding Vector vs Raster Data in Geospatial Analysis

In geospatial applications, data is primarily represented in two formats: **vector** and **raster**.  
Each format has its strengths depending on the type of spatial feature or analysis you're performing.

---

#### Vector Data

Vector data represents real-world features using geometric shapes:

- **Points**: Individual locations (e.g., trees, wells)
- **Lines**: Linear features (e.g., rivers, roads)
- **Polygons**: Areas (e.g., farm plots, districts)

Each feature can have **attributes** (e.g., crop type, owner).

```python
import geopandas as gpd

# Load a shapefile
gdf = gpd.read_file('farms.shp')

# Inspect the data
print(gdf.head())
print(gdf.geometry[0])
```

Vector data is ideal for:

- Boundary definitions
- Spatial joins and queries
- Attribute-rich analysis

---

#### Raster Data

Raster data is a grid of pixels where each pixel holds a value (e.g., reflectance, temperature, elevation).

```python
import rasterio
from rasterio.plot import show

# Load a raster image
with rasterio.open('satellite_image.tif') as src:
    print("Bands:", src.count)
    print("Size:", src.width, "x", src.height)
    show(src.read(1))  # Show first band
```

Raster data is ideal for:

- Satellite imagery
- Continuous surfaces (NDVI, elevation)
- Image classification and time series

---

#### Key Differences

| Feature      | Vector                         | Raster                        |
| ------------ | ------------------------------ | ----------------------------- |
| Structure    | Points, lines, polygons        | Grid of pixels                |
| Best For     | Boundaries, discrete objects   | Surfaces, continuous data     |
| Storage      | Geometries + attributes        | Numeric pixel values          |
| File formats | `.shp`, `.geojson`, `.gpkg`    | `.tif`, `.img`, `.nc`, `.hdf` |
| Resolution   | Based on digitization accuracy | Based on pixel size           |

---

#### Choosing the Right Format

| Task                         | Recommended Format |
| ---------------------------- | ------------------ |
| Mapping farm boundaries      | Vector             |
| Analyzing vegetation (NDVI)  | Raster             |
| Combining spatial attributes | Vector             |
| Image classification         | Raster             |

---

#### Exercises

1. Open a shapefile using GeoPandas and print the geometry type.
2. Open a raster using Rasterio and print metadata like CRS and size.
3. Plot a vector and raster layer together using `matplotlib`.
4. Describe three real-world use cases where each format is better suited.
5. Bonus: Convert NDVI raster to vector by thresholding and vectorizing high NDVI areas.

### Coordinate Reference Systems (CRS) in Geospatial Data

A **Coordinate Reference System (CRS)** defines how the two-dimensional, projected map in your GIS relates to real places on the earth.

CRS is crucial for **aligning** and **analyzing** geospatial data accurately.

---

#### Types of CRS

There are two main types of CRS:

| Type           | Description                                 | Example            |
| -------------- | ------------------------------------------- | ------------------ |
| Geographic CRS | Uses latitude and longitude (angular units) | EPSG:4326 (WGS 84) |
| Projected CRS  | Uses meters or feet on a flat surface       | EPSG:32636 (UTM)   |

---

#### Checking and Setting CRS

```python
import geopandas as gpd

# Load shapefile
gdf = gpd.read_file("farms.shp")

# Check CRS
print(gdf.crs)

# Reproject to UTM Zone 36N
gdf_utm = gdf.to_crs("EPSG:32636")
print(gdf_utm.crs)
```

You can reproject raster files too:

```python
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

with rasterio.open("satellite.tif") as src:
    dst_crs = "EPSG:4326"
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )

    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open("reprojected.tif", 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest
            )
```

---

#### Why CRS Matters

- Ensures datasets align correctly on the map
- Allows spatial measurements (distance, area)
- Required for overlays, joins, or clipping operations

---

#### Common EPSG Codes

| EPSG Code | Name                    | Description                    |
| --------- | ----------------------- | ------------------------------ |
| 4326      | WGS84                   | Global Lat/Lon (GPS)           |
| 3857      | Web Mercator            | Used in web maps (Google, OSM) |
| 32636     | UTM Zone 36N (WGS84)    | East Africa (meters)           |
| 21097     | Arc 1960 / UTM zone 37S | Kenya                          |

---

#### Exercises

1. Load a vector file and print its CRS.
2. Reproject a shapefile from WGS84 to UTM.
3. Load a raster and display its CRS.
4. Reproject a raster using Rasterio and save the output.
5. Bonus: Search the EPSG registry and find the correct code for your country’s UTM zone.

### Reading and Plotting Shapefiles in Python

Shapefiles are one of the most common formats for storing vector geospatial data.  
They typically contain **points**, **lines**, or **polygons** representing spatial features such as farms, roads, or regions.

---

#### Components of a Shapefile

A shapefile actually consists of multiple files with the same name but different extensions:

| Extension | Description                        |
| --------- | ---------------------------------- |
| `.shp`    | Geometry (points, lines, polygons) |
| `.shx`    | Shape index                        |
| `.dbf`    | Attribute table (metadata)         |
| `.prj`    | Coordinate reference system        |

You must keep these files together when loading a shapefile.

---

#### Reading a Shapefile with GeoPandas

```python
import geopandas as gpd

# Load shapefile
gdf = gpd.read_file("Nyagatare_A2021.shp")

# View first 5 rows
print(gdf.head())

# View column names
print(gdf.columns)
```

---

#### Plotting the Shapefile

```python
import matplotlib.pyplot as plt

# Simple plot
gdf.plot()
plt.title("Farm Boundaries in Nyagatare")
plt.show()

# Plot by crop type
gdf.plot(column="crop_type", legend=True, cmap="Set3")
plt.title("Crop Types in Nyagatare")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()
```

You can customize plots with colors, legends, titles, and labels.

---

#### Exploring Attribute Data

```python
# Access specific columns
print(gdf["crop_type"].value_counts())

# Filter polygons by crop type
maize_farms = gdf[gdf["crop_type"] == "Maize"]
print(maize_farms)
```

---

#### Exercises

1. Load a shapefile and print its first 5 records.
2. Plot the shapefile with default style using `.plot()`.
3. Create a color-coded plot based on the crop type or ownership.
4. Count how many features belong to each crop type.
5. Filter out polygons larger than a certain area and plot them separately.

---

#### Reproject

Use `.to_crs()` to reproject your shapefile before plotting if it appears distorted or does not align with a base map.

```python
gdf_utm = gdf.to_crs(epsg=32636)
gdf_utm.plot()
```

### Opening and Exploring Raster Files in Python

Raster data represents the world as a grid of cells (pixels), commonly used for satellite imagery, elevation models, and NDVI.

Each pixel contains a value such as reflectance, temperature, or vegetation index.

---

#### Opening a Raster File

Use the `rasterio` library to open and read raster files like `.tif`.

```python
import rasterio

# Open raster
raster_path = "nyagatare_image.tif"
with rasterio.open(raster_path) as src:
    print("Raster opened successfully!")
    print(src.profile)  # Metadata
    print("CRS:", src.crs)
    print("Width, Height:", src.width, src.height)
    print("Number of Bands:", src.count)
```

---

#### Reading Raster Bands

You can read individual bands or all bands into NumPy arrays.

```python
with rasterio.open(raster_path) as src:
    band1 = src.read(1)  # Read the first band
    print("Band shape:", band1.shape)
```

Bands represent different wavelengths or indices (e.g., Red, NIR, NDVI).

---

#### Visualizing Raster Data

Use `rasterio.plot.show()` or matplotlib to display raster images.

```python
from rasterio.plot import show
import matplotlib.pyplot as plt

with rasterio.open(raster_path) as src:
    show(src.read(1), title="Raster Band 1")

# Or manually plot with matplotlib
plt.imshow(src.read(1), cmap='viridis')
plt.title("Raster Band 1")
plt.colorbar()
plt.show()
```

---

#### Raster Metadata and Properties

Each raster contains metadata describing its structure and location.

```python
with rasterio.open(raster_path) as src:
    print("Bounds:", src.bounds)
    print("Resolution:", src.res)
    print("Data Type:", src.dtypes)
```

---

#### Exercises

1. Load a raster and print its metadata and number of bands.
2. Read and display the first band using `rasterio`.
3. Extract the pixel resolution and CRS.
4. Calculate the min and max of a raster band.
5. Try plotting the band with a different color map (`cmap`).

---

#### Tip

If you're using multi-band images (e.g., RGB or multispectral), explore combinations like:

```python
r = src.read(3)
g = src.read(2)
b = src.read(1)
rgb = np.stack([r, g, b], axis=-1)
plt.imshow(rgb / 255)
```

### Masking and Clipping Rasters by AOI

Clipping a raster by a shapefile (Area of Interest - AOI) allows you to focus your analysis on a specific region, such as a farm or district.

---

#### Required Libraries

```python
import rasterio
import geopandas as gpd
from rasterio.mask import mask
from rasterio.plot import show
```

---

#### Load the Shapefile (AOI)

```python
# Read AOI shapefile
aoi = gpd.read_file("Nyagatare_A2021.shp")
print(aoi.crs)
```

Make sure the CRS of the shapefile matches the raster.

---

#### Clip Raster by AOI

```python
with rasterio.open("nyagatare_image.tif") as src:
    # Reproject AOI to match raster CRS
    aoi = aoi.to_crs(src.crs)

    # Mask raster using shapefile geometry
    clipped_image, clipped_transform = mask(src, aoi.geometry, crop=True)
    clipped_meta = src.meta.copy()

# Update metadata
clipped_meta.update({
    "height": clipped_image.shape[1],
    "width": clipped_image.shape[2],
    "transform": clipped_transform
})

# Save clipped raster
with rasterio.open("nyagatare_clipped.tif", "w", **clipped_meta) as dst:
    dst.write(clipped_image)
```

---

#### Visualize the Clipped Raster

```python
show(clipped_image[0], title="Clipped Raster")
```

---

#### Why Clip Rasters?

- Reduce file size and processing time
- Focus analysis on relevant regions
- Prepare data for machine learning

---

#### Exercises

1. Load a shapefile and print its CRS.
2. Reproject the shapefile to match the raster CRS.
3. Use `rasterio.mask.mask` to clip the raster.
4. Save the clipped raster and view its metadata.
5. Display the clipped raster using matplotlib or rasterio.

---

#### Tips

- If your AOI has multiple polygons, consider simplifying it or clipping one feature at a time.
- You can use `.buffer(0)` on geometries to fix invalid shapes.

### Patch Extraction from Rasters

Patch extraction is a technique used in geospatial analysis and machine learning to break down large raster images into smaller, manageable chunks or **patches**.

This is useful for:

- Training machine learning models
- Managing memory and computational load
- Spatial tiling for deep learning

---

#### Required Libraries

```python
import rasterio
import numpy as np
from rasterio.windows import Window
```

---

#### Set Patch Size and Loop Through Image

```python
patch_size = 64  # Size of each patch (64x64 pixels)

with rasterio.open("nyagatare_image.tif") as src:
    for i in range(0, src.height, patch_size):
        for j in range(0, src.width, patch_size):
            window = Window(j, i, patch_size, patch_size)
            patch = src.read(window=window)
            np.save(f"patch_{i}_{j}.npy", patch)
```

This will save each patch as a `.npy` file, which can later be loaded for training or analysis.

---

#### Optional: Filter Valid Patches

You can skip patches that are full of NoData values or have too much cloud cover:

```python
if np.all(patch == 0):
    continue  # skip empty patch
```

---

#### Exercises

1. Extract 64×64 patches from a raster and save them as `.npy` files.
2. Try different patch sizes like 32, 128.
3. Count how many patches were extracted.
4. Create a mask to skip patches with low information content.
5. Visualize one patch using `matplotlib`.

```python
import matplotlib.pyplot as plt
plt.imshow(patch[0], cmap='gray')
plt.title("Sample Patch - Band 1")
plt.show()
```

---

#### Tip

If you need overlapping patches (e.g., for segmentation), reduce the stride:

```python
stride = 32  # overlap
```

This increases the number of patches and improves spatial learning in ML.

### Crop Classification with Machine Learning and Deep Learning

Crop classification is the process of identifying and labeling different crop types in satellite imagery using predictive models.

This section introduces both **Machine Learning (Random Forest)** and **Deep Learning (Convolutional Neural Networks - CNN)** approaches.

---

#### Machine Learning with Random Forest

Random Forest is an ensemble learning method that works well with tabular data extracted from raster patches.

##### Example Workflow

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load features and labels
X = np.load("features.npy")  # shape: (n_samples, n_features)
y = np.load("labels.npy")    # shape: (n_samples,)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

---

#### Deep Learning with CNN (using Keras)

CNNs are ideal for image classification. Here, we classify raster patches.

##### CNN Example

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Assume X has shape (n_samples, height, width, bands)
X = np.load("patches.npy")
y = np.load("labels.npy")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Define CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```

---

#### Model Evaluation

```python
# Evaluate accuracy
loss, accuracy = model.evaluate(X_test, y_test)
print("CNN Accuracy:", accuracy)
```

---

#### Tips

- Normalize image data before training.
- Use class weights if dataset is imbalanced.
- Save models using `model.save()` for later reuse.

---

#### Exercises

1. Train a Random Forest model using CSV or NumPy feature data.
2. Extract patches and labels to train a CNN model.
3. Compare accuracy between RF and CNN.
4. Visualize a prediction mask using matplotlib.
5. Save and reload your trained model.

### Google Earth Engine (GEE) Python API

Google Earth Engine (GEE) is a cloud-based platform for planetary-scale geospatial analysis. Its Python API allows users to access GEE datasets and run computations using Python scripts.

---

#### Setup and Authentication

Install Earth Engine Python API:

```bash
pip install earthengine-api
```

Authenticate and initialize:

```python
import ee
ee.Authenticate()
ee.Initialize()
```

---

#### Load and Filter a Dataset

Example: Load Sentinel-2 imagery for a region and time range.

```python
point = ee.Geometry.Point([30.05, -1.95])  # Example location

collection = (
    ee.ImageCollection('COPERNICUS/S2_SR')
    .filterBounds(point)
    .filterDate('2021-01-01', '2021-03-31')
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
)

image = collection.median()
```

---

#### Export Image to Google Drive

```python
task = ee.batch.Export.image.toDrive(
    image=image,
    description='S2_Nyagatare_2021Q1',
    folder='GEE_exports',
    fileNamePrefix='sentinel2_2021',
    region=point.buffer(10000).bounds().getInfo()['coordinates'],
    scale=10,
    crs='EPSG:4326'
)
task.start()
```

---

#### Visualize with Folium

```python
import folium
from geemap import foliumap

Map = folium.Map(location=[-1.95, 30.05], zoom_start=10)
Map.add_child(folium.TileLayer('Stamen Terrain'))

vis_params = {'min': 0, 'max': 3000, 'bands': ['B4', 'B3', 'B2']}
Map.add_ee_layer(image, vis_params, 'Sentinel-2')
Map
```

> Note: You need `geemap` for `add_ee_layer`.

```bash
pip install geemap
```

---

#### Tips

- Use `.first()` to inspect the first image in a collection.
- Combine with `geopandas` for geospatial integration.
- Use the Code Editor (code.earthengine.google.com) to validate expressions.

---

#### Exercises

1. Load Landsat-8 images for a region of interest and filter by date/clouds.
2. Export an NDVI image to Google Drive.
3. Visualize an image using Folium and geemap.
4. Use `.mean()` instead of `.median()` and compare results.

---

#### Resources

- [GEE Python API Docs](https://developers.google.com/earth-engine/guides/python_install)
- [Geemap Docs](https://geemap.org/)

### Creating and Managing Python Environments

Managing Python environments is essential for avoiding package conflicts and keeping your projects isolated. Below are common tools and methods for environment management.

---

#### Using `venv` (Built-in Python Virtual Environment)

```bash
# Create environment
python -m venv myenv

# Activate
source myenv/bin/activate      # Linux/macOS
myenv\Scripts\activate.bat   # Windows

# Deactivate
deactivate
```

Install packages inside the environment:

```bash
pip install numpy pandas
```

---

#### Using `virtualenv`

A more flexible tool, especially for older Python versions.

```bash
pip install virtualenv
virtualenv myenv
source myenv/bin/activate
```

---

#### Using `conda` (Anaconda or Miniforge)

```bash
# Create a new environment
conda create -n crop_env python=3.10

# Activate
conda activate crop_env

# Install packages
conda install numpy pandas rasterio

# Deactivate
conda deactivate
```

##### Miniforge

Lightweight alternative to Anaconda with conda-forge as default channel:

```bash
# Download and install Miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

---

#### Export and Reuse Environment

Export environment:

```bash
conda env export > environment.yml
```

Recreate environment from file:

```bash
conda env create -f environment.yml
```

---

#### Using `pip-tools` for better dependency control

```bash
pip install pip-tools

# Create requirements
echo "rasterio" > requirements.in
pip-compile requirements.in
pip install -r requirements.txt
```

---

#### Tips

- Use separate environments per project.
- Prefer `conda` for geospatial libraries (e.g., rasterio, geopandas).
- Save `requirements.txt` or `environment.yml` in your repo for reproducibility.

---

#### Exercises

1. Create a `venv` or `conda` environment for your crop mapping project.
2. Install `rasterio`, `numpy`, `geopandas`, and test import.
3. Export the environment and share with a colleague.
