# Data Loading Functions Documentation

This document explains all the functions used to load and parse MNIST and Fashion MNIST datasets. These functions handle reading the binary .idx file format and converting it into usable data structures.

## Table of Contents
- [Data Structures](#data-structures)
- [File Format](#file-format)
- [Loading Functions](#loading-functions)
- [Memory Management](#memory-management)
- [Utility Functions](#utility-functions)

## Data Structures

### pixel_data
```c
typedef struct pixel_data {
    uint8_t** neuron_activation;  // 2D array of pixel data
    uint32_t size;                // Number of images
    uint32_t rows;                // Image height
    uint32_t cols;                // Image width
} pixel_data;
```

**Purpose:** Stores all the image data from MNIST/Fashion MNIST datasets.

**How it works:**
- `neuron_activation[i]` points to the pixel data for image `i`
- Each image is stored as a 1D array of bytes
- `size` tells you how many images are in the dataset
- `rows` and `cols` give the dimensions of each image

## File Format

### MNIST/Fashion MNIST .idx File Format

The datasets use a binary format with the following structure:

**Image Files:**
```
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number (MSB first)
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
```

**Label Files:**
```
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
```

**Important Notes:**
- All integers are stored in big-endian format
- Magic numbers identify the file type
- Pixel values range from 0-255 (grayscale)
- Labels are single bytes representing class (0-9)

## Loading Functions

### get_image_pixel_data
```c
struct pixel_data* get_image_pixel_data(FILE* file);
```

**What it does:** Reads and parses an MNIST/Fashion MNIST image file.

**Parameters:**
- `file`: File pointer to an open .idx image file

**Returns:** Pointer to a `pixel_data` structure containing all images

**How it works:**
1. Reads and validates the magic number (should be 0x00000803)
2. Reads the number of images, rows, and columns
3. Converts big-endian integers to little-endian
4. Allocates memory for all images
5. Reads pixel data for each image
6. Returns the complete dataset

**Example:**
```c
FILE* file = fopen("fashion-mnist/train-images-idx3-ubyte", "rb");
struct pixel_data* data = get_image_pixel_data(file);
fclose(file);

// Now you can access:
// data->size = number of images
// data->rows = image height (28)
// data->cols = image width (28)
// data->neuron_activation[i] = pixel data for image i
```

**Memory:** Allocates memory for the structure and all image data

### get_image_labels
```c
unsigned char* get_image_labels(FILE* file);
```

**What it does:** Reads and parses an MNIST/Fashion MNIST label file.

**Parameters:**
- `file`: File pointer to an open .idx label file

**Returns:** Array of unsigned chars containing all labels

**How it works:**
1. Reads and validates the magic number (should be 0x00000801)
2. Reads the number of labels
3. Converts big-endian integer to little-endian
4. Allocates memory for all labels
5. Reads all label bytes
6. Returns the label array

**Example:**
```c
FILE* file = fopen("fashion-mnist/train-labels-idx1-ubyte", "rb");
unsigned char* labels = get_image_labels(file);
fclose(file);

// Now you can access:
// labels[i] = label for image i (0-9)
```

**Memory:** Allocates memory for the label array

## Memory Management

### image_data_finalizer
```c
void image_data_finalizer(struct pixel_data* data);
```

**What it does:** Frees all memory allocated for image data.

**Parameters:**
- `data`: Pointer to pixel_data structure to free

**How it works:**
1. Frees each individual image array
2. Frees the array of image pointers
3. Frees the main structure
4. Sets pointers to NULL to prevent use-after-free

**Example:**
```c
struct pixel_data* data = get_image_pixel_data(file);
// ... use the data ...
image_data_finalizer(data);  // Clean up memory
```

**Important:** Always call this when done with image data to prevent memory leaks

### image_label_finalizer
```c
void image_label_finalizer(unsigned char* label_array);
```

**What it does:** Frees memory allocated for label data.

**Parameters:**
- `label_array`: Pointer to label array to free

**How it works:**
1. Frees the label array
2. Sets pointer to NULL

**Example:**
```c
unsigned char* labels = get_image_labels(file);
// ... use the labels ...
image_label_finalizer(labels);  // Clean up memory
```

**Important:** Always call this when done with label data to prevent memory leaks

## Utility Functions

### big_to_little_endian
```c
uint32_t big_to_little_endian(uint32_t value);
```

**What it does:** Converts a 32-bit integer from big-endian to little-endian format.

**Parameters:**
- `value`: 32-bit integer in big-endian format

**Returns:** 32-bit integer in little-endian format

**How it works:**
1. Extracts each byte from the big-endian value
2. Reconstructs the value in little-endian order
3. Returns the converted value

**Example:**
```c
uint32_t big_endian = 0x12345678;
uint32_t little_endian = big_to_little_endian(big_endian);
// little_endian = 0x78563412
```

**Usage:** Called internally by the loading functions to handle byte order conversion

## Complete Usage Example

Here's how to load a complete dataset:

```c
// Load training images
FILE* image_file = fopen("fashion-mnist/train-images-idx3-ubyte", "rb");
if (image_file == NULL) {
    perror("Failed to open image file");
    return 1;
}

struct pixel_data* train_images = get_image_pixel_data(image_file);
fclose(image_file);

// Load training labels
FILE* label_file = fopen("fashion-mnist/train-labels-idx1-ubyte", "rb");
if (label_file == NULL) {
    perror("Failed to open label file");
    image_data_finalizer(train_images);
    return 1;
}

unsigned char* train_labels = get_image_labels(label_file);
fclose(label_file);

// Now you can use the data
printf("Loaded %u training images\n", train_images->size);
printf("Image dimensions: %ux%u\n", train_images->rows, train_images->cols);

// Access individual images and labels
for (uint32_t i = 0; i < train_images->size; i++) {
    uint8_t* image_pixels = train_images->neuron_activation[i];
    uint8_t label = train_labels[i];
    
    printf("Image %u: label = %u\n", i, label);
    
    // Access pixel at position (row, col)
    int row = 14, col = 14;
    uint8_t pixel = image_pixels[row * train_images->cols + col];
    printf("Pixel at (%d,%d): %u\n", row, col, pixel);
}

// Clean up when done
image_data_finalizer(train_images);
image_label_finalizer(train_labels);
```

## Error Handling

### Common Errors and Solutions

**"Failed to open file"**
- Check if the file path is correct
- Ensure the file exists in the specified location
- Verify file permissions

**"Failed to read magic number"**
- File might be corrupted or in wrong format
- Check if you're reading the right file type
- Verify file is not empty

**"Not enough bytes read"**
- File might be truncated or corrupted
- Check if the file size matches expected size
- Verify the file is complete

**Memory allocation failures**
- System might be out of memory
- Try reducing batch size or closing other applications
- Check for memory leaks in your code

### Error Checking Best Practices

```c
// Always check file operations
FILE* file = fopen("data/file.idx", "rb");
if (file == NULL) {
    perror("Failed to open file");
    return 1;
}

// Check data loading
struct pixel_data* data = get_image_pixel_data(file);
if (data == NULL) {
    perror("Failed to load image data");
    fclose(file);
    return 1;
}

// Verify data integrity
if (data->size == 0 || data->rows == 0 || data->cols == 0) {
    printf("Invalid data dimensions\n");
    image_data_finalizer(data);
    fclose(file);
    return 1;
}
```

## Performance Considerations

### Memory Usage
- **Training set**: ~47MB (60,000 images × 784 pixels)
- **Test set**: ~8MB (10,000 images × 784 pixels)
- **Labels**: ~60KB (60,000 bytes)

### Loading Speed
- Binary format is very fast to read
- No parsing or conversion overhead
- Limited by disk I/O speed

### Optimization Tips
1. **Load once**: Load datasets once at startup and reuse
2. **Batch access**: Access images in batches for better cache performance
3. **Memory mapping**: For very large datasets, consider memory mapping
4. **Compression**: Files can be compressed to save disk space

## Dataset Information

### MNIST Dataset
- **Training images**: 60,000
- **Test images**: 10,000
- **Image size**: 28×28 pixels
- **Classes**: 10 (digits 0-9)
- **File sizes**: ~47MB training, ~8MB test

### Fashion MNIST Dataset
- **Training images**: 60,000
- **Test images**: 10,000
- **Image size**: 28×28 pixels
- **Classes**: 10 (fashion categories)
- **File sizes**: ~47MB training, ~8MB test

### Fashion MNIST Classes
- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot

## File Naming Convention

The expected file structure:
```
fashion-mnist/
├── train-images-idx3-ubyte    # Training images
├── train-labels-idx1-ubyte    # Training labels
├── t10k-images-idx3-ubyte     # Test images
└── t10k-labels-idx1-ubyte     # Test labels
```

**Note:** The files must be downloaded separately as they are not included in the repository due to size.

