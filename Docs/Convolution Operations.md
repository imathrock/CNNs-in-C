# Convolution Operations Documentation

This document explains all the convolution and pooling functions in the CNN implementation. These functions handle the core image processing operations that make convolutional neural networks work.

## Table of Contents
- [Image2D Structure](#image2d-structure)
- [Image Creation Functions](#image-creation-functions)
- [Convolution Functions](#convolution-functions)
- [Pooling Functions](#pooling-functions)
- [Utility Functions](#utility-functions)

## Image2D Structure

The `Image2D` structure is the main data type for handling images and feature maps:

```c
typedef struct {
    int rows;           // Number of rows in the image
    int cols;           // Number of columns in the image
    float *Data;        // Pixel data stored as 1D array
    int* maxidx;        // Indices for max pooling (can be NULL)
} Image2D;
```

**How it works:**
- Images are stored as 1D arrays for better memory access
- Pixel at position (i,j) is accessed as `Data[i * cols + j]`
- `maxidx` stores the positions of maximum values for unpooling operations

## Image Creation Functions

### CreateImage
```c
Image2D CreateImage(int rows, int cols);
```

**What it does:** Creates a new image with the specified dimensions.

**Parameters:**
- `rows`: Number of rows (height)
- `cols`: Number of columns (width)

**Returns:** A new Image2D structure with allocated memory

**Example:**
```c
Image2D img = CreateImage(28, 28);  // Creates a 28x28 image
```

**Memory:** Allocates memory for both pixel data and max pooling indices

### CreateKernel
```c
Image2D CreateKernel(int rows, int cols);
```

**What it does:** Creates a convolution kernel with random weights.

**Parameters:**
- `rows`: Kernel height (usually 3, 5, or 7)
- `cols`: Kernel width (usually 3, 5, or 7)

**Returns:** A new kernel with random weights initialized using He initialization

**Example:**
```c
Image2D kernel = CreateKernel(5, 5);  // Creates a 5x5 kernel
```

**Note:** Kernels don't need maxidx, so it's set to NULL

### CreateConvImg
```c
Image2D CreateConvImg(Image2D img, Image2D kernel);
```

**What it does:** Creates an image with the correct size for storing convolution results.

**Parameters:**
- `img`: Input image
- `kernel`: Convolution kernel

**Returns:** A new image sized for convolution output

**Size calculation:** `(img.rows - kernel.rows + 1) × (img.cols - kernel.cols + 1)`

### CreatePoolImg
```c
Image2D CreatePoolImg(Image2D img, int ker_size, int stride);
```

**What it does:** Creates an image with the correct size for storing pooling results.

**Parameters:**
- `img`: Input image
- `ker_size`: Size of pooling window (usually 2)
- `stride`: Stride of pooling operation (usually 2)

**Returns:** A new image sized for pooling output

**Size calculation:** `((img.rows - ker_size) / stride + 1) × ((img.cols - ker_size) / stride + 1)`

## Convolution Functions

### Conv2D
```c
void Conv2D(Image2D Kernel, Image2D image, Image2D convimg);
```

**What it does:** Performs 2D convolution of an image with a kernel.

**Parameters:**
- `Kernel`: Convolution kernel (filter)
- `image`: Input image
- `convimg`: Output image (must be pre-allocated with correct size)

**How it works:**
1. Slides the kernel over the input image
2. At each position, multiplies kernel values with corresponding image pixels
3. Sums all products to get one output pixel
4. Repeats for all valid positions

**Example:**
```c
Image2D input = CreateImage(28, 28);
Image2D kernel = CreateKernel(5, 5);
Image2D output = CreateConvImg(input, kernel);
Conv2D(kernel, input, output);
```

**Important:** The output image must be pre-allocated with the correct size!

### backprop_kernel
```c
void backprop_kernel(Image2D delta_kernel, Image2D Kernel, Image2D Unpooled, Image2D Image);
```

**What it does:** Computes gradients for kernel weights during backpropagation.

**Parameters:**
- `delta_kernel`: Gradient buffer for kernel updates
- `Kernel`: Current kernel weights
- `Unpooled`: Gradient from the next layer
- `Image`: Original input image

**How it works:**
1. Takes the gradient from the next layer
2. Computes how much each kernel weight contributed to the error
3. Accumulates gradients in `delta_kernel`

**Usage:** Called during backpropagation to update kernel weights

## Pooling Functions

### MAXPOOL
```c
void MAXPOOL(Image2D image, Image2D poolimg, int ker_size, int stride);
```

**What it does:** Performs max pooling on an image.

**Parameters:**
- `image`: Input image
- `poolimg`: Output pooled image (must be pre-allocated)
- `ker_size`: Size of pooling window (usually 2)
- `stride`: Stride of pooling operation (usually 2)

**How it works:**
1. Divides the image into non-overlapping windows
2. Finds the maximum value in each window
3. Stores the maximum value and its position
4. Output image is smaller than input

**Example:**
```c
Image2D input = CreateImage(24, 24);
Image2D output = CreatePoolImg(input, 2, 2);
MAXPOOL(input, output, 2, 2);  // Results in 12x12 image
```

### MAXUNPOOL
```c
void MAXUNPOOL(Image2D unpooled, Image2D pooled);
```

**What it does:** Reverses max pooling operation during backpropagation.

**Parameters:**
- `unpooled`: Output image (larger size)
- `pooled`: Input pooled image (smaller size)

**How it works:**
1. Takes the gradient from the next layer
2. Places gradients back at the positions where maximum values were found
3. Fills other positions with zeros

**Usage:** Used during backpropagation to compute gradients for pooling layers

## Utility Functions

### ImageInput
```c
void ImageInput(Image2D image, uint8_t* Data);
```

**What it does:** Loads pixel data from a byte array into an Image2D structure.

**Parameters:**
- `image`: Target Image2D structure
- `Data`: Source byte array (usually from MNIST files)

**How it works:**
1. Converts each byte to a float
2. Normalizes values to range [0, 1] by dividing by 255
3. Stores in the image's Data array

**Example:**
```c
uint8_t pixel_data[784];  // 28x28 = 784 pixels
Image2D img = CreateImage(28, 28);
ImageInput(img, pixel_data);
```

### ImageReLU
```c
void ImageReLU(Image2D image);
```

**What it does:** Applies ReLU activation function to all pixels in an image.

**How it works:**
- If pixel value > 0: keep the value
- If pixel value ≤ 0: set to 0

**Formula:** `f(x) = max(0, x)`

### kernel_update
```c
void kernel_update(Image2D delta_kernel, Image2D Kernel, float learning_rate);
```

**What it does:** Updates kernel weights using computed gradients.

**Parameters:**
- `delta_kernel`: Accumulated gradients
- `Kernel`: Kernel to update
- `learning_rate`: Learning rate for gradient descent

**How it works:**
1. Multiplies gradients by learning rate
2. Subtracts from current kernel weights
3. Uses SIMD instructions for speed

**Formula:** `Kernel = Kernel - learning_rate * delta_kernel`

### zero_kernel
```c
void zero_kernel(Image2D Kernel);
```

**What it does:** Sets all kernel weights to zero.

**Usage:** Called to reset gradient buffers between batches

## Memory Management

**Important Notes:**
- All `Create*` functions allocate memory that must be freed
- `Conv2D` and `MAXPOOL` require pre-allocated output images
- Always check that output images have the correct size before calling functions
- Use `free()` to release memory when done with images

**Example cleanup:**
```c
Image2D img = CreateImage(28, 28);
// ... use the image ...
free(img.Data);
free(img.maxidx);
```

## Performance Tips

1. **Reuse Images:** Create images once and reuse them instead of creating new ones for each operation
2. **Correct Sizing:** Always ensure output images have the correct dimensions
3. **SIMD Optimization:** The `kernel_update` function uses AVX instructions for faster computation
4. **Memory Layout:** Images use 1D arrays for better cache performance

## Common Mistakes

1. **Wrong Output Size:** Not allocating output images with correct dimensions
2. **Memory Leaks:** Forgetting to free allocated images
3. **Null Pointers:** Not checking if memory allocation succeeded
4. **Incorrect Strides:** Using wrong stride values in pooling operations
