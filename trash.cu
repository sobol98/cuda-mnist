#include <stdio.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


// Kernel Sobela dla kierunku X
__constant__ float d_kernel_x[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

// Kernel Sobela dla kierunku Y
__constant__ float d_kernel_y[9] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

__global__ void sobelFilter(
    unsigned char* input,
    unsigned char* output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float gx = 0.0f;
        float gy = 0.0f;

        // Oblicz konwolucję 3x3
        for(int ky = -1; ky <= 1; ky++) {
            for(int kx = -1; kx <= 1; kx++) {
                int ix = x + kx;
                int iy = y + ky;
                
                // Sprawdź granice obrazu
                if(ix >= 0 && ix < width && iy >= 0 && iy < height) {
                    float pixel = input[iy * width + ix];
                    int k = (ky + 1) * 3 + (kx + 1);
                    gx += pixel * d_kernel_x[k];
                    gy += pixel * d_kernel_y[k];
                }
            }
        }

        // Oblicz magnitudę gradientu
        float magnitude = sqrtf(gx * gx + gy * gy);
        
        // Normalizuj do zakresu 0-255
        output[y * width + x] = min(max((int)magnitude, 0), 255);
    }
}

int main(int argc, char** argv) {
    if(argc != 3) {
        printf("Użycie: %s <input.jpg> <output.png>\n", argv[0]);
        return 1;
    }

    // Wczytaj obraz
    int width, height, channels;
    unsigned char* h_input = stbi_load(argv[1], &width, &height, &channels, 0);
    if(!h_input) {
        printf("Błąd wczytywania obrazu\n");
        return -1;
    }

    printf("Wczytano obraz: %dx%d, %d kanałów\n", width, height, channels);

    // Konwertuj do skali szarości jeśli obraz jest kolorowy
    unsigned char* h_gray = (unsigned char*)malloc(width * height);
    for(int i = 0; i < width * height; i++) {
        h_gray[i] = (h_input[i * channels] + 
                     h_input[i * channels + 1] + 
                     h_input[i * channels + 2]) / 3;
    }

    // Alokuj pamięć na GPU
    unsigned char *d_input, *d_output;
    cudaMalloc(&d_input, width * height);
    cudaMalloc(&d_output, width * height);

    // Kopiuj dane na GPU
    cudaMemcpy(d_input, h_gray, width * height, cudaMemcpyHostToDevice);

    // Konfiguracja wykonania kernela
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);

    // Uruchom konwolucję
    sobelFilter<<<gridSize, blockSize>>>(d_input, d_output, width, height);

    // Sprawdź błędy
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Pobierz wynik
    unsigned char* h_output = (unsigned char*)malloc(width * height);
    cudaMemcpy(h_output, d_output, width * height, cudaMemcpyDeviceToHost);

    // Zapisz wynik
    stbi_write_png(argv[2], width, height, 1, h_output, width);

    // Zwolnij pamięć
    stbi_image_free(h_input);
    free(h_gray);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    printf("Zapisano przetworzone zdjęcie do: %s\n", argv[2]);
    return 0;
}





__global__ void maxpool2d(
    const float *input, 
    float *output,
    int batch_size,
    int channels,
    int input_size,
    int pool_size
) {
    int b = blockIdx.z;    // batch index
    int c = blockIdx.y;    // channel
    int h = blockIdx.x / (input_size/pool_size);  // output height
    int w = blockIdx.x % (input_size/pool_size);  // output width
    
    if (b < batch_size && c < channels && 
        h < input_size/pool_size && w < input_size/pool_size) {
        
        float max_val = -INFINITY;
        
        // Find maximum in pool window
        for (int ph = 0; ph < pool_size; ph++) {
            for (int pw = 0; pw < pool_size; pw++) {
                int in_h = h * pool_size + ph;
                int in_w = w * pool_size + pw;
                int idx = ((b * channels + c) * input_size + in_h) * input_size + in_w;
                max_val = fmaxf(max_val, input[idx]);
            }
        }
        
        // Save output
        int out_size = input_size / pool_size;
        int out_idx = ((b * channels + c) * out_size + h) * out_size + w;
        output[out_idx] = max_val;
    }
}

__global__ void linear(
    const float *input,
    float *output,
    const float *weights,
    const float *bias,
    int batch_size,
    int in_features,
    int out_features
) {
    int b = blockIdx.y;    // batch index
    int o = blockIdx.x * blockDim.x + threadIdx.x;  // output feature
    
    if (b < batch_size && o < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < in_features; i++) {
            sum += input[b * in_features + i] * weights[o * in_features + i];
        }
        sum += bias[o];
        output[b * out_features + o] = sum;
    }
}

__global__ void softmax(float *input, float *output, int batch_size, int num_classes) {
    int b = blockIdx.x;    // batch index
    
    if (b < batch_size) {
        float max_val = -INFINITY;
        float sum = 0.0f;
        
        // Find max value
        for (int i = 0; i < num_classes; i++) {
            max_val = fmaxf(max_val, input[b * num_classes + i]);
        }
        
        // Compute exp and sum
        for (int i = 0; i < num_classes; i++) {
            float exp_val = expf(input[b * num_classes + i] - max_val);
            output[b * num_classes + i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for (int i = 0; i < num_classes; i++) {
            output[b * num_classes + i] /= sum;
        }
    }
}

// Inicjalizacja parametrów sieci
void initializeNetwork(NetworkParams &params, int batch_size) {
    // Conv1
    cudaMalloc(&params.conv1_weights, 32 * 1 * 3 * 3 * sizeof(float));
    cudaMalloc(&params.conv1_bias, 32 * sizeof(float));
    cudaMalloc(&params.conv1_output, batch_size * 32 * 26 * 26 * sizeof(float));
    
    // Conv2
    cudaMalloc(&params.conv2_weights, 64 * 32 * 3 * 3 * sizeof(float));
    cudaMalloc(&params.conv2_bias, 64 * sizeof(float));
    cudaMalloc(&params.conv2_output, batch_size * 64 * 24 * 24 * sizeof(float));
    
    // Pool output
    cudaMalloc(&params.pool_output, batch_size * 64 * 12 * 12 * sizeof(float));
    
    // FC1
    cudaMalloc(&params.fc1_weights, 9216 * 128 * sizeof(float));
    cudaMalloc(&params.fc1_bias, 128 * sizeof(float));
    cudaMalloc(&params.fc1_output, batch_size * 128 * sizeof(float));
    
    // FC2
    cudaMalloc(&params.fc2_weights, 128 * 10 * sizeof(float));
    cudaMalloc(&params.fc2_bias, 10 * sizeof(float));
    cudaMalloc(&params.fc2_output, batch_size * 10 * sizeof(float));
}

void freeNetwork(NetworkParams &params) {
    cudaFree(params.conv1_weights);
    cudaFree(params.conv1_bias);
    cudaFree(params.conv1_output);
    cudaFree(params.conv2_weights);
    cudaFree(params.conv2_bias);
    cudaFree(params.conv2_output);
    cudaFree(params.pool_output);
    cudaFree(params.fc1_weights);
    cudaFree(params.fc1_bias);
    cudaFree(params.fc1_output);
    cudaFree(params.fc2_weights);
    cudaFree(params.fc2_bias);
    cudaFree(params.fc2_output);
}

// Funkcja forward pass
void forward(float *input, NetworkParams &params, int batch_size) {
    dim3 conv1_grid(IMG_SIZE * IMG_SIZE, 32, batch_size);
    conv2d<<<conv1_grid, 1>>>(input, params.conv1_output, params.conv1_weights, 
                             params.conv1_bias, batch_size, 1, 32, IMG_SIZE, 3);
    
    dim3 conv2_grid(26 * 26, 64, batch_size);
    conv2d<<<conv2_grid, 1>>>(params.conv1_output, params.conv2_output, 
                             params.conv2_weights, params.conv2_bias, 
                             batch_size, 32, 64, 26, 3);
    
    dim3 pool_grid(12 * 12, 64, batch_size);
    maxpool2d<<<pool_grid, 1>>>(params.conv2_output, params.pool_output, 
                               batch_size, 64, 24, 2);
    
    dim3 fc1_grid(128, batch_size);
    linear<<<fc1_grid, 1>>>(params.pool_output, params.fc1_output, 
                           params.fc1_weights, params.fc1_bias, 
                           batch_size, 9216, 128);
    
    dim3 fc2_grid(10, batch_size);
    linear<<<fc2_grid, 1>>>(params.fc1_output, params.fc2_output, 
                           params.fc2_weights, params.fc2_bias, 
                           batch_size, 128, 10);
    
    softmax<<<batch_size, 1>>>(params.fc2_output, params.fc2_output, 
                              batch_size, 10);
}



// Reszta kodu (main, read_mnist_images, read_mnist_labels) pozostaje bez zmian