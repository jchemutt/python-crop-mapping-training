
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

```python
x = 5
name = 'Maize'
is_crop = True

area = 10
yield_per_hectare = 2.5
total_yield = area * yield_per_hectare
print('Total yield:', total_yield)

value = '25'
value_int = int(value)
print(value_int + 5)
```

**Exercises:**
1. Create variables for a crop name, area of land, and yield.
2. Compute total expected yield.
3. Convert string to integer.
4. Use conditionals to check if yield > 20.

### Lists, Dictionaries, and Tuples

```python
# List
crops = ['Maize', 'Beans', 'Sorghum']
print(crops[0])
crops.append('Cassava')

# Tuple
coordinates = (1.95, 30.06)
print(coordinates[0])

# Dictionary
crop_yields = {'Maize': 2.5, 'Beans': 1.8}
print(crop_yields['Maize'])
crop_yields['Sorghum'] = 2.0
```

### Loops and Conditionals

```python
# If-Else
if yield_per_hectare > 3:
    print('High yield')
elif yield_per_hectare > 2:
    print('Moderate yield')
else:
    print('Low yield')

# For loop
for crop in crops:
    print('Processing crop:', crop)

# While loop
counter = 0
while counter < 3:
    print('Count:', counter)
    counter += 1

# Break & Continue
for crop in crops:
    if crop == 'Beans':
        continue
    print(crop)
```

### Functions and File I/O

```python
# Function
def calculate_yield(area, rate):
    return area * rate

# Read file
with open('crop_names.txt', 'r') as file:
    for line in file:
        print('Crop:', line.strip())

# Write file
with open('output.txt', 'w') as file:
    file.write('Crop Yield Report\n')
    file.write('Maize: 2.5 tons/ha')
```

## Working with Geospatial Data

### Vector vs Raster Data

- **Vector**: Points, lines, polygons
- **Raster**: Gridded pixels, e.g., reflectance, elevation

```python
import geopandas as gpd
gdf = gpd.read_file('farms.shp')
print(gdf.geometry[0])

import rasterio
from rasterio.plot import show
with rasterio.open('satellite_image.tif') as src:
    show(src.read(1))
```

### Coordinate Reference Systems (CRS)

```python
# Shapefile CRS
gdf = gpd.read_file('Nyagatare_A2021.shp')
print(gdf.crs)
gdf = gdf.to_crs('EPSG:32636')

# Raster CRS
with rasterio.open('nyagatare_image.tif') as src:
    print(src.crs)
```

### Reading and Plotting Shapefiles

```python
import matplotlib.pyplot as plt
gdf.plot(column='crop_type', legend=True, figsize=(10, 6))
plt.title('Crop Types in Nyagatare')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
```

## Raster Data Handling

### Opening and Exploring Rasters

```python
with rasterio.open('nyagatare_image.tif') as src:
    print(src.meta)
    show(src.read(1))
```

### Masking and Clipping

```python
from rasterio.mask import mask
shapefile = gpd.read_file('Nyagatare_A2021.shp')
with rasterio.open('nyagatare_image.tif') as src:
    clipped, transform = mask(src, shapefile.geometry, crop=True)
    plt.imshow(clipped[0])
```

## Patch Extraction

```python
from rasterio.windows import Window
with rasterio.open('nyagatare_image.tif') as src:
    for i in range(0, src.height, 64):
        for j in range(0, src.width, 64):
            patch = src.read(window=Window(j, i, 64, 64))
            np.save(f'patch_{i}_{j}.npy', patch)
```

## Simple Crop Classification: ML and DL

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
X = np.load('features.npy')
y = np.load('labels.npy')
model = RandomForestClassifier()
model.fit(X, y)
```

### CNN Model

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10)
```

---

This Markdown version is structured, clean, and portable for training sites, Jupyter Notebooks, or GitHub repos.
