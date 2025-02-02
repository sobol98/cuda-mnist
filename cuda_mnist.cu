#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Constants for the network architecture
#define INPUT_SIZE 784  // 28x28
#define CONV1_CHANNELS 32
#define CONV2_CHANNELS 64
#define FC1_SIZE 128
#define NUM_CLASSES 10
#define BATCH_SIZE 64
#define LEARNING_RATE 0.01f

// Structure for the CNN layers
struct ConvLayer {
    float* weights;
    float* bias;
    int in_channels;
    int out_channels;
    int kernel_size;
};

struct FCLayer {
    float* weights;
    float* bias;
    int in_features;
    int out_features;
};

// Network structure
struct Network {
    ConvLayer conv1;
    ConvLayer conv2;
    FCLayer fc1;
    FCLayer fc2;
};

// CUDA kernel for convolution operation
__global__ void conv2d_kernel(float* input, float* output, float* weights, float* bias,
                             int batch_size, int in_channels, int out_channels,
                             int height, int width, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_channels * height * width) return;

    int w_idx = idx % width;
    int h_idx = (idx / width) % height;
    int c_out = (idx / (width * height)) % out_channels;
    int b_idx = idx / (width * height * out_channels);

    float sum = 0.0f;
    for (int c_in = 0; c_in < in_channels; c_in++) {
        for (int kh = 0; kh < kernel_size; kh++) {
            for (int kw = 0; kw < kernel_size; kw++) {
                int h_in = h_idx + kh - kernel_size/2;
                int w_in = w_idx + kw - kernel_size/2;
                
                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width) {
                    int input_idx = ((b_idx * in_channels + c_in) * height + h_in) * width + w_in;
                    int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    output[idx] = sum + bias[c_out];
}

// CUDA kernel for ReLU activation
__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

// CUDA kernel for max pooling
__global__ void max_pool2d_kernel(float* input, float* output,
                                 int batch_size, int channels,
                                 int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_height = height/2;
    int out_width = width/2;
    
    if (idx >= batch_size * channels * out_height * out_width) return;

    int w_out = idx % out_width;
    int h_out = (idx / out_width) % out_height;
    int c = (idx / (out_width * out_height)) % channels;
    int b = idx / (out_width * out_height * channels);

    float maxVal = -INFINITY;
    for (int h = 0; h < 2; h++) {
        for (int w = 0; w < 2; w++) {
            int h_in = h_out * 2 + h;
            int w_in = w_out * 2 + w;
            int in_idx = ((b * channels + c) * height + h_in) * width + w_in;
            maxVal = fmaxf(maxVal, input[in_idx]);
        }
    }
    output[idx] = maxVal;
}

// Initialize network parameters
void init_network(Network& net) {
    // Conv1 initialization
    cudaMalloc(&net.conv1.weights, sizeof(float) * CONV1_CHANNELS * 1 * 3 * 3);
    cudaMalloc(&net.conv1.bias, sizeof(float) * CONV1_CHANNELS);
    net.conv1.in_channels = 1;
    net.conv1.out_channels = CONV1_CHANNELS;
    net.conv1.kernel_size = 3;

    // Conv2 initialization
    cudaMalloc(&net.conv2.weights, sizeof(float) * CONV2_CHANNELS * CONV1_CHANNELS * 3 * 3);
    cudaMalloc(&net.conv2.bias, sizeof(float) * CONV2_CHANNELS);
    net.conv2.in_channels = CONV1_CHANNELS;
    net.conv2.out_channels = CONV2_CHANNELS;
    net.conv2.kernel_size = 3;

    // FC1 initialization
    cudaMalloc(&net.fc1.weights, sizeof(float) * 9216 * FC1_SIZE);
    cudaMalloc(&net.fc1.bias, sizeof(float) * FC1_SIZE);
    net.fc1.in_features = 9216;
    net.fc1.out_features = FC1_SIZE;

    // FC2 initialization
    cudaMalloc(&net.fc2.weights, sizeof(float) * FC1_SIZE * NUM_CLASSES);
    cudaMalloc(&net.fc2.bias, sizeof(float) * NUM_CLASSES);
    net.fc2.in_features = FC1_SIZE;
    net.fc2.out_features = NUM_CLASSES;

    // Initialize weights using cuRAND (not shown for brevity)
}

// Forward pass
void forward(Network& net, float* input, float* output, int batch_size) {
    // Allocate temporary buffers
    float *conv1_out, *conv2_out, *pool_out, *fc1_out;
    cudaMalloc(&conv1_out, sizeof(float) * batch_size * CONV1_CHANNELS * 26 * 26);
    cudaMalloc(&conv2_out, sizeof(float) * batch_size * CONV2_CHANNELS * 24 * 24);
    cudaMalloc(&pool_out, sizeof(float) * batch_size * CONV2_CHANNELS * 12 * 12);
    cudaMalloc(&fc1_out, sizeof(float) * batch_size * FC1_SIZE);

    // Conv1 + ReLU
    dim3 conv1_blocks((batch_size * CONV1_CHANNELS * 26 * 26 + 255) / 256);
    dim3 conv1_threads(256);
    conv2d_kernel<<<conv1_blocks, conv1_threads>>>(
        input, conv1_out, net.conv1.weights, net.conv1.bias,
        batch_size, 1, CONV1_CHANNELS, 28, 28, 3
    );
    relu_kernel<<<conv1_blocks, conv1_threads>>>(conv1_out, batch_size * CONV1_CHANNELS * 26 * 26);

    // Conv2 + ReLU
    dim3 conv2_blocks((batch_size * CONV2_CHANNELS * 24 * 24 + 255) / 256);
    dim3 conv2_threads(256);
    conv2d_kernel<<<conv2_blocks, conv2_threads>>>(
        conv1_out, conv2_out, net.conv2.weights, net.conv2.bias,
        batch_size, CONV1_CHANNELS, CONV2_CHANNELS, 26, 26, 3
    );
    relu_kernel<<<conv2_blocks, conv2_threads>>>(conv2_out, batch_size * CONV2_CHANNELS * 24 * 24);

    // Max pooling
    dim3 pool_blocks((batch_size * CONV2_CHANNELS * 12 * 12 + 255) / 256);
    dim3 pool_threads(256);
    max_pool2d_kernel<<<pool_blocks, pool_threads>>>(
        conv2_out, pool_out, batch_size, CONV2_CHANNELS, 24, 24
    );

    // FC layers (using cuBLAS for matrix multiplication)
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    // FC1
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                FC1_SIZE, batch_size, 9216,
                &alpha,
                net.fc1.weights, FC1_SIZE,
                pool_out, 9216,
                &beta,
                fc1_out, FC1_SIZE);
    relu_kernel<<<(batch_size * FC1_SIZE + 255) / 256, 256>>>(fc1_out, batch_size * FC1_SIZE);

    // FC2
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                NUM_CLASSES, batch_size, FC1_SIZE,
                &alpha,
                net.fc2.weights, NUM_CLASSES,
                fc1_out, FC1_SIZE,
                &beta,
                output, NUM_CLASSES);

    // Cleanup
    cublasDestroy(handle);
    cudaFree(conv1_out);
    cudaFree(conv2_out);
    cudaFree(pool_out);
    cudaFree(fc1_out);
}

// // Main training loop
// int main() {
//     // Initialize network
//     Network net;
//     init_network(net);

//     // Training loop would go here
//     // You would need to:
//     // 1. Load MNIST data
//     // 2. Create mini-batches
//     // 3. Forward pass
//     // 4. Calculate loss
//     // 5. Backward pass (gradient calculation)
//     // 6. Update weights
//     // 7. Repeat for specified number of epochs


//     // to run nvcc -o cuda_mnist cuda_mnist.cu -lcublas -lcurand
//     // what is lcublas?
//     // what is lcurand?

//     return 0;
// }




// Main training loop
// Simple cross entropy loss kernel
__global__ void cross_entropy_loss_kernel(float* predictions, int* targets, float* loss,
                                        int batch_size, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        if (i == targets[idx]) {
            sum += -logf(predictions[idx * num_classes + i]);
        }
    }
    atomicAdd(loss, sum / batch_size);
}

// Structure for MNIST data
struct MNISTData {
    unsigned char* images;
    unsigned char* labels;
    int num_images;
};

// Load MNIST data from file
MNISTData load_mnist(const char* image_filename, const char* label_filename) {
    MNISTData data;
    FILE *fp;
    
    // Load images
    fp = fopen(image_filename, "rb");
    if (!fp) {
        printf("Failed to open image file\n");
        exit(1);
    }
    
    int magic_number, num_images, num_rows, num_cols;
    fread(&magic_number, sizeof(int), 1, fp);
    fread(&num_images, sizeof(int), 1, fp);
    fread(&num_rows, sizeof(int), 1, fp);
    fread(&num_cols, sizeof(int), 1, fp);
    
    // Reverse byte order if necessary
    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);
    
    data.num_images = num_images;
    data.images = (unsigned char*)malloc(num_images * num_rows * num_cols);
    fread(data.images, 1, num_images * num_rows * num_cols, fp);
    fclose(fp);
    
    // Load labels
    fp = fopen(label_filename, "rb");
    if (!fp) {
        printf("Failed to open label file\n");
        exit(1);
    }
    
    fread(&magic_number, sizeof(int), 1, fp);
    fread(&num_images, sizeof(int), 1, fp);
    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    
    data.labels = (unsigned char*)malloc(num_images);
    fread(data.labels, 1, num_images, fp);
    fclose(fp);
    
    return data;
}

int main() {
    // 1. Initialize network and load MNIST data
    Network net;
    init_network(net);
    
    MNISTData train_data = load_mnist("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    printf("Loaded %d training images\n", train_data.num_images);
    
    // Allocate GPU memory for training data
    float *d_train_images, *d_train_labels;
    cudaMalloc(&d_train_images, train_data.num_images * 784 * sizeof(float));
    cudaMalloc(&d_train_labels, train_data.num_images * sizeof(int));
    
    // Convert and copy data to GPU
    float* h_train_images = (float*)malloc(train_data.num_images * 784 * sizeof(float));
    for (int i = 0; i < train_data.num_images * 784; i++) {
        h_train_images[i] = train_data.images[i] / 255.0f;
    }
    cudaMemcpy(d_train_images, h_train_images, train_data.num_images * 784 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_labels, train_data.labels, train_data.num_images * sizeof(int), cudaMemcpyHostToDevice);
    
    // Training parameters
    const int num_epochs = 10;
    const int num_batches = train_data.num_images / BATCH_SIZE;
    
    // Allocate memory for intermediate results
    float *d_output, *d_loss;
    cudaMalloc(&d_output, BATCH_SIZE * NUM_CLASSES * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));
    
    // 7. Training loop (epochs)
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float total_loss = 0.0f;
        
        // 2. Process mini-batches
        for (int batch = 0; batch < num_batches; batch++) {
            int offset = batch * BATCH_SIZE;
            float* batch_images = d_train_images + (offset * 784);
            int* batch_labels = (int*)d_train_labels + offset;
            
            // 3. Forward pass
            forward(net, batch_images, d_output, BATCH_SIZE);
            
            // 4. Calculate loss
            float batch_loss = 0.0f;
            cudaMemcpy(d_loss, &batch_loss, sizeof(float), cudaMemcpyHostToDevice);
            cross_entropy_loss_kernel<<<(BATCH_SIZE + 255) / 256, 256>>>(
                d_output, batch_labels, d_loss, BATCH_SIZE, NUM_CLASSES);
            cudaMemcpy(&batch_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
            total_loss += batch_loss;
            
            // 5. Backward pass (simplified - just printing loss for now)
            // Note: Full implementation would need gradient calculation
            
            // 6. Update weights (simplified)
            // Note: Full implementation would use optimizer like SGD
            
            if (batch % 100 == 0) {
                printf("Epoch %d/%d, Batch %d/%d, Loss: %f\n", 
                       epoch + 1, num_epochs, batch, num_batches, batch_loss);
            }
        }
        
        printf("Epoch %d/%d completed, Average Loss: %f\n", 
               epoch + 1, num_epochs, total_loss / num_batches);
    }
    
    // Cleanup
    cudaFree(d_train_images);
    cudaFree(d_train_labels);
    cudaFree(d_output);
    cudaFree(d_loss);
    free(h_train_images);
    free(train_data.images);
    free(train_data.labels);
    
    return 0;
}