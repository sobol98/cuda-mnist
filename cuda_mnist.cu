/*
* nvcc -o cuda_mnist cuda_mnist.cu -lcurand
* ./cuda_mnist
*
*
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define IMG_SIZE 28
#define FILTER_SIZE 3
#define POOL_SIZE 2
#define FC1_INPUT 9216
#define FC1_OUTPUT 128
#define FC2_OUTPUT 10
#define BATCH_SIZE 128
#define EPOCHS 1
#define LEARNING_RATE 0.001f  // Zamiast 1.0f
#define GAMMA 0.7f


struct NetworkParams{
    // Conv1: 1->32 channels
    float *conv1_weights;    // [in_channels=1, out_channels=32, kernel_size=3, stride=1]
    float *conv1_bias;       // [out_channels=32]
    float *conv1_output;     // [batch_size, out_channels=32, height=26, width=26]
    
    // Conv2: 32->64 channels
    float *conv2_weights;    // [in_channels=32, out_channels=64, kernel_size=3, stride=1]
    float *conv2_bias;       // [out_channels=64]
    float *conv2_output;     // [batch_size, out_channels=64, height=24, width=24]
    
    // Pooling output
    float *pool_output;      // [batch_size, channels=64, height=12, width=12]
    
    // FC1: 9216->128
    float *fc1_weights;      // [in_features=9216, out_features=128]
    float *fc1_bias;         // [out_features=128]
    float *fc1_output;       // [batch_size, out_features=128]
    
    // FC2: 128->10
    float *fc2_weights;      // [in_features=128, out_features=10]
    float *fc2_bias;         // [out_features=10]
    float *fc2_output;       // [batch_size, out_features=10]
};

struct DropoutStates {
    curandState* states1;
    curandState* states2;
    float* mask1;
    float* mask2;
};

struct DetailedResult {
    int true_label;
    int pred_label;
    std::vector<float> probabilities;
};

struct NetworkGradients {
    float *conv1_weights_grad;
    float *conv1_bias_grad;
    float *conv1_output_grad;  // Dodane
    float *conv1_input_grad;   

    float *conv2_weights_grad;
    float *conv2_bias_grad;
    float *conv2_output_grad;  // Dodane
    float *conv2_input_grad;   

    float *pool_output_grad;   

    float *fc1_weights_grad;
    float *fc1_bias_grad;
    float *fc1_output_grad;    // Dodane
    float *fc1_input_grad;     

    float *fc2_weights_grad;
    float *fc2_bias_grad;
    float *fc2_output_grad;    
    float *fc2_input_grad;     

    float *input_grad;         
};


void read_mnist_images(const std::string &file, std::vector<std::vector<float>> &images) {
    std::ifstream f(file, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "error: " << file << std::endl;
        exit(1);
    }
    
    int magic, num_images, rows, cols;
    f.read(reinterpret_cast<char *>(&magic), 4);
    f.read(reinterpret_cast<char *>(&num_images), 4);
    f.read(reinterpret_cast<char *>(&rows), 4);
    f.read(reinterpret_cast<char *>(&cols), 4);

    // endian conversion
    magic = __builtin_bswap32(magic);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    images.resize(num_images, std::vector<float>(rows * cols));

    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < rows * cols; j++) {
            unsigned char pixel;
            f.read(reinterpret_cast<char *>(&pixel), 1);
            images[i][j] = pixel / 255.0f;  // (0-1 values), normalization
        }
    }
    f.close();
}

void read_mnist_labels(const std::string &file, std::vector<int> &labels) {
    std::ifstream f(file, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Błąd otwierania pliku: " << file << std::endl;
        exit(1);
    }
    
    int magic, num_labels;
    f.read(reinterpret_cast<char *>(&magic), 4);
    f.read(reinterpret_cast<char *>(&num_labels), 4);

    magic = __builtin_bswap32(magic);
    num_labels = __builtin_bswap32(num_labels);

    labels.resize(num_labels);

    for (int i = 0; i < num_labels; i++) {
        unsigned char label;
        f.read(reinterpret_cast<char *>(&label), 1);
        labels[i] = label;
    }
    f.close();
}


__global__ void conv2d(const float* input, float* output, const float* weights, const float* bias, int batch_size, int in_channels, int out_channels, int input_size, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;  // batch index

    int output_size = input_size - kernel_size + 1;
    if (x >= output_size || y >= output_size || b >= batch_size) {
        return;
    }

    for (int oc = 0; oc < out_channels; oc++) {
        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ic++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_y = y + ky;
                    int in_x = x + kx;
                    int in_idx = ((b * in_channels + ic) * input_size + in_y) * input_size + in_x;
                    
                    int w_idx = ((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx;
                    sum += input[in_idx] * weights[w_idx];
                }
            }
        }

        sum += bias[oc]; //bias
        sum = fmaxf(0.0f, sum); //ReLU

        int out_idx = ((b * out_channels + oc) * output_size + y) * output_size + x;
        output[out_idx] = sum;
    }
}


__global__ void maxpool2d(const float* input, float* output, int batch_size, int channels, int input_size, int pool_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;  // batch index

    int output_size = input_size / pool_size;
    if (x >= output_size || y >= output_size || b >= batch_size) {
        return;
    }

    for (int c = 0; c < channels; c++) {
        float max_val = -INFINITY;

        for (int py = 0; py < pool_size; py++) {
            for (int px = 0; px < pool_size; px++) {
                int in_y = y * pool_size + py;
                int in_x = x * pool_size + px;
                
                int in_idx = ((b * channels + c) * input_size + in_y) * input_size + in_x;
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
        int out_idx = ((b * channels + c) * output_size + y) * output_size + x;
        output[out_idx] = max_val;
    }
}

__global__ void linear(const float* input, float* output, const float* weights, const float* bias, int batch_size, int in_features, int out_features) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;  // batch index

    if (x >= out_features || b >= batch_size) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < in_features; i++) {
        sum += input[b * in_features + i] * weights[x * in_features + i];
    }
    sum += bias[x];
    output[b * out_features + x] = sum;
}


__global__ void log_softmax(float *input, float *output, int batch_size, int num_classes) {
    int b = blockIdx.x;    // batch index
    
    if (b < batch_size) {
        float max_val = -INFINITY;
        float sum = 0.0f;
        
        for (int i = 0; i < num_classes; i++) {
            max_val = fmaxf(max_val, input[b * num_classes + i]);
        }
        
        for (int i = 0; i < num_classes; i++) {
            float exp_val = expf(input[b * num_classes + i] - max_val);
            output[b * num_classes + i] = exp_val;
            sum += exp_val;
        }
        
        for (int i = 0; i < num_classes; i++) {
            output[b * num_classes + i] = logf(output[b * num_classes + i] / sum);
        }
    }
}


__global__ void dropout(float* data, float* mask, float dropout_prob, int size, int batch_size, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;  // batch index
    
    if (idx < size && b < batch_size) {
        int index = b * size + idx;
        float rand = curand_uniform(&states[index]);
        mask[index] = rand > dropout_prob ? 1.0f : 0.0f;
        data[index] = data[index] * mask[index] / (1.0f - dropout_prob);
    }
}

void initialize_weights(NetworkParams &params) {
    // Inicjalizacja wag małymi losowymi wartościami
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(NULL));  // Użyj aktualnego czasu jako ziarna

    float conv1_std = sqrtf(2.0f / (1 * 3 * 3 + 32));
    float conv2_std = sqrtf(2.0f / (32 * 3 * 3 + 64));
    float fc1_std = sqrtf(2.0f / (9216 + 128));
    float fc2_std = sqrtf(2.0f / (128 + 10));

    // Conv1 weights
    curandGenerateNormal(gen, params.conv1_weights, 1 * 32 * 3 * 3, 0.0f, conv1_std);
    curandGenerateNormal(gen, params.conv1_bias, 32, 0.0f, conv1_std);

    // Conv2 weights
    curandGenerateNormal(gen, params.conv2_weights, 32 * 64 * 3 * 3, 0.0f, conv2_std);
    curandGenerateNormal(gen, params.conv2_bias, 64, 0.0f, conv2_std);

    // FC1 weights
    curandGenerateNormal(gen, params.fc1_weights, 9216 * 128, 0.0f, fc1_std);
    curandGenerateNormal(gen, params.fc1_bias, 128, 0.0f, fc1_std);

    // FC2 weights
    curandGenerateNormal(gen, params.fc2_weights, 128 * 10, 0.0f, fc2_std);
    curandGenerateNormal(gen, params.fc2_bias, 10, 0.0f, fc2_std);

    curandDestroyGenerator(gen);
}


__global__ void nll_loss(float* pred, int* target, float* loss, int batch_size, int num_classes) {
    __shared__ float total_loss;
    
    if (threadIdx.x == 0) {
        total_loss = 0.0f;
        if (blockIdx.x == 0) {
            printf("nll_loss first prediction values: ");
            for (int i = 0; i < num_classes; i++) {
                printf("%.4f ", pred[i]);
            }
            printf("\n");
        }
    }
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < batch_size) {
        int t = target[tid];
        if (t >= 0 && t < num_classes) {
            float pred_val = pred[tid * num_classes + t];
            // Dodajemy clipping dla numerycznej stabilności
            pred_val = fmaxf(fminf(pred_val, 0.0f), -20.0f); // dla log_softmax wartości są ujemne
            atomicAdd(&total_loss, -pred_val);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(loss, total_loss / batch_size);
    }
}




// Kernel do aktualizacji wag z gradientem i learning rate
__global__ void update_weights(
    float* weights, float* grads, float* biases, float* bias_grads, 
    int size, float learning_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * grads[idx];
        if (biases && bias_grads) {
            biases[idx] -= learning_rate * bias_grads[idx];
        }
    }
}

void optimizer_step(NetworkParams &params, NetworkGradients &grads, float learning_rate) {
    dim3 blockSize(256);

    // Conv1 weights update
    dim3 conv1_grid((1 * 32 * 3 * 3 + blockSize.x - 1) / blockSize.x);
    update_weights<<<conv1_grid, blockSize>>>(
        params.conv1_weights, grads.conv1_weights_grad, 
        params.conv1_bias, grads.conv1_bias_grad, 
        1 * 32 * 3 * 3, learning_rate);

    // Conv2 weights update
    dim3 conv2_grid((32 * 64 * 3 * 3 + blockSize.x - 1) / blockSize.x);
    update_weights<<<conv2_grid, blockSize>>>(
        params.conv2_weights, grads.conv2_weights_grad, 
        params.conv2_bias, grads.conv2_bias_grad, 
        32 * 64 * 3 * 3, learning_rate);

    // FC1 weights update
    dim3 fc1_grid((9216 * 128 + blockSize.x - 1) / blockSize.x);
    update_weights<<<fc1_grid, blockSize>>>(
        params.fc1_weights, grads.fc1_weights_grad, 
        params.fc1_bias, grads.fc1_bias_grad, 
        9216 * 128, learning_rate);

    // FC2 weights update
    dim3 fc2_grid((128 * 10 + blockSize.x - 1) / blockSize.x);
    update_weights<<<fc2_grid, blockSize>>>(
        params.fc2_weights, grads.fc2_weights_grad, 
        params.fc2_bias, grads.fc2_bias_grad, 
        128 * 10, learning_rate);
}

void zero_grad(NetworkGradients &grads, int batch_size) {
    // Conv1 gradienty
    cudaMemset(grads.conv1_weights_grad, 0, 1 * 32 * 3 * 3 * sizeof(float));
    cudaMemset(grads.conv1_bias_grad, 0, 32 * sizeof(float));
    cudaMemset(grads.conv1_output_grad, 0, batch_size * 32 * 26 * 26 * sizeof(float));
    cudaMemset(grads.conv1_input_grad, 0, batch_size * 1 * 28 * 28 * sizeof(float));

    // Conv2 gradienty
    cudaMemset(grads.conv2_weights_grad, 0, 32 * 64 * 3 * 3 * sizeof(float));
    cudaMemset(grads.conv2_bias_grad, 0, 64 * sizeof(float));
    cudaMemset(grads.conv2_output_grad, 0, batch_size * 64 * 24 * 24 * sizeof(float));
    cudaMemset(grads.conv2_input_grad, 0, batch_size * 32 * 26 * 26 * sizeof(float));

    // Pool gradient
    cudaMemset(grads.pool_output_grad, 0, batch_size * 64 * 12 * 12 * sizeof(float));

    // FC1 gradienty
    cudaMemset(grads.fc1_weights_grad, 0, 9216 * 128 * sizeof(float));
    cudaMemset(grads.fc1_bias_grad, 0, 128 * sizeof(float));
    cudaMemset(grads.fc1_output_grad, 0, batch_size * 128 * sizeof(float));
    cudaMemset(grads.fc1_input_grad, 0, batch_size * 9216 * sizeof(float));

    // FC2 gradienty
    cudaMemset(grads.fc2_weights_grad, 0, 128 * 10 * sizeof(float));
    cudaMemset(grads.fc2_bias_grad, 0, 10 * sizeof(float));
    cudaMemset(grads.fc2_output_grad, 0, batch_size * 10 * sizeof(float));
    cudaMemset(grads.fc2_input_grad, 0, batch_size * 128 * sizeof(float));

    cudaMemset(grads.input_grad, 0, batch_size * IMG_SIZE * IMG_SIZE * sizeof(float));
}

void initializeNetwork(NetworkParams &params, int batch_size) {
    // Conv1
    cudaMalloc(&params.conv1_weights, 1* 32 * 3 * 3 * sizeof(float));
    cudaMalloc(&params.conv1_bias, 32 * sizeof(float));
    cudaMalloc(&params.conv1_output, batch_size * 32 * 26 * 26 * sizeof(float));
    
    // Conv2
    cudaMalloc(&params.conv2_weights, 32* 64 * 3 * 3 * sizeof(float));
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


void forward(float* input, NetworkParams &params, DropoutStates &dropout_states, int batch_size, bool is_training) {
    dim3 blockSize(16, 16);
    
    // Conv1 + ReLU
    dim3 conv1_grid((IMG_SIZE + blockSize.x - 1) / blockSize.x,
                    (IMG_SIZE + blockSize.y - 1) / blockSize.y,
                    batch_size);
    conv2d<<<conv1_grid, blockSize>>>(input, params.conv1_output, 
                                     params.conv1_weights, params.conv1_bias,
                                     batch_size, 1, 32, IMG_SIZE, 3);

    // Conv2 + ReLU
    dim3 conv2_grid((26 + blockSize.x - 1) / blockSize.x,
                    (26 + blockSize.y - 1) / blockSize.y,
                    batch_size);
    conv2d<<<conv2_grid, blockSize>>>(params.conv1_output, params.conv2_output,
                                     params.conv2_weights, params.conv2_bias,
                                     batch_size, 32, 64, 26, 3);

    // MaxPool
    dim3 pool_grid((24 + blockSize.x - 1) / blockSize.x,
                   (24 + blockSize.y - 1) / blockSize.y,
                   batch_size);
    maxpool2d<<<pool_grid, blockSize>>>(params.conv2_output, params.pool_output,
                                       batch_size, 64, 24, 2);

    if (is_training) {
        // Dropout1 (0.25)
        dim3 dropout1_grid((9216 + 255) / 256, batch_size);
        dropout<<<dropout1_grid, 256>>>(params.pool_output, dropout_states.mask1,
                                      0.25f, 9216, batch_size, dropout_states.states1);
    }

    // FC1 + ReLU
    dim3 fc1_grid((128 + blockSize.x - 1) / blockSize.x, 1, batch_size);
    linear<<<fc1_grid, blockSize>>>(params.pool_output, params.fc1_output,
                                   params.fc1_weights, params.fc1_bias,
                                   batch_size, 9216, 128);

    if (is_training) {
        // Dropout2 (0.5)
        dim3 dropout2_grid((128 + 255) / 256, batch_size);
        dropout<<<dropout2_grid, 256>>>(params.fc1_output, dropout_states.mask2,
                                      0.5f, 128, batch_size, dropout_states.states2);
    }

    // FC2
    dim3 fc2_grid((10 + blockSize.x - 1) / blockSize.x, 1, batch_size);
    linear<<<fc2_grid, blockSize>>>(params.fc1_output, params.fc2_output,
                                   params.fc2_weights, params.fc2_bias,
                                   batch_size, 128, 10);

    // Log_softmax
    dim3 softmax_grid(batch_size);
    log_softmax<<<softmax_grid, 1>>>(params.fc2_output, params.fc2_output,
                                    batch_size, 10);
}


void initializeDropout(DropoutStates &dropout_states, int batch_size) {
    cudaMalloc(&dropout_states.states1, batch_size * 9216 * sizeof(curandState));
    cudaMalloc(&dropout_states.states2, batch_size * 128 * sizeof(curandState));
    cudaMalloc(&dropout_states.mask1, batch_size * 9216 * sizeof(float));
    cudaMalloc(&dropout_states.mask2, batch_size * 128 * sizeof(float));
}


// ---------------------- BACKWARD ----------------------

__global__ void softmax_backward(float* output_grad, float* log_softmax_output, 
                               int* targets, float* input_grad, 
                               int batch_size, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size) return;
    
    // Dodaj debugowanie
    if (b == 0) {
        printf("softmax_backward: output_grad=%p, log_softmax_output=%p, targets=%p, input_grad=%p\n",
               output_grad, log_softmax_output, targets, input_grad);
    }
    
    // Dodaj synchronizację wątków
    __syncthreads();
    
    for (int c = 0; c < num_classes; c++) {
        int idx = b * num_classes + c;
        if (targets[b] < 0 || targets[b] >= num_classes) continue;  // Dodaj sprawdzenie zakresu
        
        float exp_val = expf(log_softmax_output[idx]);
        input_grad[idx] = (c == targets[b]) ? (1.0f - exp_val) : (-exp_val);
    }
}

__global__ void maxpool_backward(const float* input, const float* output, 
                               float* grad_output, float* grad_input,
                               int batch_size, int channels, int input_size, int pool_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    int output_size = input_size / pool_size;
    if (x >= output_size || y >= output_size || b >= batch_size) return;

    for (int c = 0; c < channels; c++) {
        float max_val = -INFINITY;
        int max_idx_x = -1, max_idx_y = -1;

        // Znajdź pozycję maksymalnej wartości w oknie poolingu
        for (int py = 0; py < pool_size; py++) {
            for (int px = 0; px < pool_size; px++) {
                int in_y = y * pool_size + py;
                int in_x = x * pool_size + px;
                
                int in_idx = ((b * channels + c) * input_size + in_y) * input_size + in_x;
                float val = input[in_idx];
                
                if (val > max_val) {
                    max_val = val;
                    max_idx_x = in_x;
                    max_idx_y = in_y;
                }
            }
        }

        // Propaguj gradient tylko do pozycji maksymalnej wartości
        int out_idx = ((b * channels + c) * output_size + y) * output_size + x;
        float grad = grad_output[out_idx];
        
        if (max_idx_x != -1 && max_idx_y != -1) {
            int grad_idx = ((b * channels + c) * input_size + max_idx_y) * input_size + max_idx_x;
            atomicAdd(&grad_input[grad_idx], grad);
        }
    }
}

// Poprawiony kernel fc_backward
__global__ void fc_backward(float* prev_layer, float* grad_output, 
                          float* grad_weights, float* grad_bias, 
                          float* weights, float* input_grad, 
                          int batch_size, int in_features, int out_features) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    
    if (x < out_features && b < batch_size) {
        float local_grad = grad_output[b * out_features + x];
        
        // Gradient biasu
        if (local_grad != 0.0f) {
            atomicAdd(&grad_bias[x], local_grad);
        }
        
        // Gradient wag i wejścia
        for (int i = 0; i < in_features; i++) {
            float prev_val = prev_layer[b * in_features + i];
            if (prev_val != 0.0f) {
                atomicAdd(&grad_weights[x * in_features + i], prev_val * local_grad);
            }
            atomicAdd(&input_grad[b * in_features + i], weights[x * in_features + i] * local_grad);
        }
    }
}

// Poprawiony kernel relu_backward
__global__ void relu_backward(float* input, float* grad_output, 
                            float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

// Poprawiony kernel conv2d_backward
__global__ void conv2d_backward(float* input, float* grad_output, 
                              float* grad_weights, float* grad_bias,
                              float* weights, float* grad_input,
                              int batch_size, int in_channels, int out_channels, 
                              int input_size, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z;

    int output_size = input_size - kernel_size + 1;
    if (x >= output_size || y >= output_size || b >= batch_size) return;

    for (int oc = 0; oc < out_channels; oc++) {
        float grad_val = grad_output[((b * out_channels + oc) * output_size + y) * output_size + x];
        
        // Gradient biasu
        atomicAdd(&grad_bias[oc], grad_val);

        for (int ic = 0; ic < in_channels; ic++) {
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int in_y = y + ky;
                    int in_x = x + kx;
                    
                    if (in_y >= input_size || in_x >= input_size) continue;
                    
                    int in_idx = ((b * in_channels + ic) * input_size + in_y) * input_size + in_x;
                    int w_idx = ((oc * in_channels + ic) * kernel_size + ky) * kernel_size + kx;

                    atomicAdd(&grad_weights[w_idx], input[in_idx] * grad_val);
                    atomicAdd(&grad_input[in_idx], weights[w_idx] * grad_val);
                }
            }
        }
    }
}

// Poprawiona funkcja backward
void backward(NetworkParams &params, NetworkGradients &grads, int* d_labels, float* input) {
    cudaError_t err;
    

    printf("Checking pointers in backward:\n");
    printf("params.fc2_output: %p\n", params.fc2_output);
    printf("grads.fc2_output_grad: %p\n", grads.fc2_output_grad);
    printf("d_labels: %p\n", d_labels);
    
    // Sprawdź czy pierwsza wartość d_labels jest poprawna
    int first_label;
    cudaMemcpy(&first_label, d_labels, sizeof(int), cudaMemcpyDeviceToHost);
    printf("First label: %d\n", first_label);


    // Softmax backward
    dim3 softmax_block(256);
    dim3 softmax_grid((BATCH_SIZE + softmax_block.x - 1) / softmax_block.x);
    softmax_backward<<<softmax_grid, 1>>>(
        params.fc2_output, params.fc2_output, d_labels, 
        grads.fc2_output_grad, BATCH_SIZE, 10
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in softmax_backward: %s\n", cudaGetErrorString(err));
    }

    // FC2 backward
    dim3 fc2_block(256);
    dim3 fc2_grid((10 + fc2_block.x - 1) / fc2_block.x, BATCH_SIZE);
    fc_backward<<<fc2_grid, fc2_block>>>(
        params.fc1_output, grads.fc2_output_grad, 
        grads.fc2_weights_grad, grads.fc2_bias_grad,
        params.fc2_weights, grads.fc1_input_grad,
        BATCH_SIZE, 128, 10
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in fc2_backward: %s\n", cudaGetErrorString(err));
    }

    // ReLU FC1 backward
    dim3 relu_fc1_block(256);
    dim3 relu_fc1_grid((128 * BATCH_SIZE + relu_fc1_block.x - 1) / relu_fc1_block.x);
    relu_backward<<<relu_fc1_grid, relu_fc1_block>>>(
        params.fc1_output, grads.fc1_input_grad, 
        grads.fc1_input_grad, 128 * BATCH_SIZE
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in relu_fc1_backward: %s\n", cudaGetErrorString(err));
    }

    // FC1 backward
    dim3 fc1_block(256);
    dim3 fc1_grid((128 + fc1_block.x - 1) / fc1_block.x, BATCH_SIZE);
    fc_backward<<<fc1_grid, fc1_block>>>(
        params.pool_output, grads.fc1_input_grad, 
        grads.fc1_weights_grad, grads.fc1_bias_grad,
        params.fc1_weights, grads.pool_output_grad,
        BATCH_SIZE, 9216, 128
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in fc1_backward: %s\n", cudaGetErrorString(err));
    }

    dim3 pool_block(16, 16);
    dim3 pool_grid((24 + pool_block.x - 1) / pool_block.x,
                (24 + pool_block.y - 1) / pool_block.y,
                BATCH_SIZE);
    maxpool_backward<<<pool_grid, pool_block>>>(
        params.conv2_output,        // input
        params.pool_output,         // output
        grads.pool_output_grad,     // grad_output
        grads.conv2_output_grad,    // grad_input
        BATCH_SIZE, 64, 24, 2
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in maxpool_backward: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();


    // Conv2 backward
    dim3 conv_block(16, 16);
    dim3 conv2_grid((24 + conv_block.x - 1) / conv_block.x, 
                    (24 + conv_block.y - 1) / conv_block.y, 
                    BATCH_SIZE);
    conv2d_backward<<<conv2_grid, conv_block>>>(
        params.conv1_output, grads.pool_output_grad, 
        grads.conv2_weights_grad, grads.conv2_bias_grad,
        params.conv2_weights, grads.conv2_input_grad,
        BATCH_SIZE, 32, 64, 26, 3
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in conv2_backward: %s\n", cudaGetErrorString(err));
    }

    // ReLU Conv2 backward
    dim3 relu_conv2_block(256);
    dim3 relu_conv2_grid((64 * 24 * 24 * BATCH_SIZE + relu_conv2_block.x - 1) / relu_conv2_block.x);
    relu_backward<<<relu_conv2_grid, relu_conv2_block>>>(
        params.conv2_output, grads.conv2_input_grad, 
        grads.conv2_input_grad, 64 * 24 * 24 * BATCH_SIZE
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in relu_conv2_backward: %s\n", cudaGetErrorString(err));
    }

    // Conv1 backward
    dim3 conv1_grid((26 + conv_block.x - 1) / conv_block.x, 
                    (26 + conv_block.y - 1) / conv_block.y, 
                    BATCH_SIZE);
    conv2d_backward<<<conv1_grid, conv_block>>>(
        input, grads.conv2_input_grad, 
        grads.conv1_weights_grad, grads.conv1_bias_grad,
        params.conv1_weights, grads.conv1_input_grad,
        BATCH_SIZE, 1, 32, 28, 3
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in conv1_backward: %s\n", cudaGetErrorString(err));
    }

    // ReLU Conv1 backward
    dim3 relu_conv1_block(256);
    dim3 relu_conv1_grid((32 * 26 * 26 * BATCH_SIZE + relu_conv1_block.x - 1) / relu_conv1_block.x);
    relu_backward<<<relu_conv1_grid, relu_conv1_block>>>(
        params.conv1_output, grads.conv1_output_grad, 
        grads.conv1_output_grad, 32 * 26 * 26 * BATCH_SIZE
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in relu_conv1_backward: %s\n", cudaGetErrorString(err));
    }

    // Synchronizacja na końcu backward pass
    cudaDeviceSynchronize();
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

void freeGradients(NetworkGradients &grads) {
    // Conv1 gradienty
    cudaFree(grads.conv1_weights_grad);
    cudaFree(grads.conv1_bias_grad);
    cudaFree(grads.conv1_output_grad);
    cudaFree(grads.conv1_input_grad);

    // Conv2 gradienty
    cudaFree(grads.conv2_weights_grad);
    cudaFree(grads.conv2_bias_grad);
    cudaFree(grads.conv2_output_grad);
    cudaFree(grads.conv2_input_grad);

    // Pool gradient
    cudaFree(grads.pool_output_grad);

    // FC1 gradienty
    cudaFree(grads.fc1_weights_grad);
    cudaFree(grads.fc1_bias_grad);
    cudaFree(grads.fc1_output_grad);
    cudaFree(grads.fc1_input_grad);

    // FC2 gradienty
    cudaFree(grads.fc2_weights_grad);
    cudaFree(grads.fc2_bias_grad);
    cudaFree(grads.fc2_output_grad);
    cudaFree(grads.fc2_input_grad);

    cudaFree(grads.input_grad);
}

void train(std::vector<std::vector<float>> &train_images, std::vector<int> &train_labels) {
    NetworkParams params;
    DropoutStates dropout_states;
    NetworkGradients grads;
    
    // Initialize network and dropout states
    initializeNetwork(params, BATCH_SIZE);
    initializeDropout(dropout_states, BATCH_SIZE);
    initialize_weights(params);

    int num_batches = train_images.size() / BATCH_SIZE;
    
    // Allocate device memory for input batch and labels
    float *d_input;
    int *d_labels;
    cudaMalloc(&d_input, BATCH_SIZE * IMG_SIZE * IMG_SIZE * sizeof(float));    
    cudaMalloc(&d_labels, BATCH_SIZE * sizeof(int));

    cudaMalloc(&grads.input_grad, BATCH_SIZE * IMG_SIZE * IMG_SIZE * sizeof(float));
    
    cudaMalloc(&grads.conv1_weights_grad, 1 * 32 * 3 * 3 * sizeof(float));
    cudaMalloc(&grads.conv1_bias_grad, 32 * sizeof(float));

    cudaMalloc(&grads.conv2_weights_grad, 32 * 64 * 3 * 3 * sizeof(float));
    cudaMalloc(&grads.conv2_bias_grad, 64 * sizeof(float));

    cudaMalloc(&grads.fc1_weights_grad, 9216 * 128 * sizeof(float));
    cudaMalloc(&grads.fc1_bias_grad, 128 * sizeof(float));

    cudaMalloc(&grads.fc2_weights_grad, 128 * 10 * sizeof(float));
    cudaMalloc(&grads.fc2_bias_grad, 10 * sizeof(float));

    cudaMalloc(&grads.fc2_input_grad, BATCH_SIZE * 128 * sizeof(float));
    cudaMalloc(&grads.fc2_output_grad, BATCH_SIZE * 10 * sizeof(float));

    cudaMalloc(&grads.fc1_input_grad, BATCH_SIZE * 9216 * sizeof(float));
    cudaMalloc(&grads.fc1_output_grad, BATCH_SIZE * 128 * sizeof(float));

    cudaMalloc(&grads.conv1_input_grad, BATCH_SIZE * 1 * 28 * 28 * sizeof(float));
    cudaMalloc(&grads.conv1_output_grad, BATCH_SIZE * 32 * 26 * 26 * sizeof(float));

    cudaMalloc(&grads.conv2_input_grad, BATCH_SIZE * 32 * 26 * 26 * sizeof(float));
    cudaMalloc(&grads.conv2_output_grad, BATCH_SIZE * 64 * 24 * 24 * sizeof(float));
    
    cudaMalloc(&grads.pool_output_grad, BATCH_SIZE * 64 * 12 * 12 * sizeof(float));


    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        float total_loss = 0.0f;
        
        for (int batch = 0; batch < num_batches; ++batch) {
            // Copy batch data to device
            std::vector<float> batch_images(BATCH_SIZE * IMG_SIZE * IMG_SIZE);
            std::vector<int> batch_labels(BATCH_SIZE);
            
            for (int i = 0; i < BATCH_SIZE; ++i) {
                std::copy(train_images[batch * BATCH_SIZE + i].begin(), 
                          train_images[batch * BATCH_SIZE + i].end(), 
                          batch_images.begin() + i * IMG_SIZE * IMG_SIZE);
                batch_labels[i] = train_labels[batch * BATCH_SIZE + i];
                if (batch_labels[i] < 0 || batch_labels[i] >= 10) {
                    printf("Warning: Invalid label %d at batch %d, index %d\n", batch_labels[i], batch, i);
                }
            }
            
            cudaMemcpy(d_input, batch_images.data(), 
                       BATCH_SIZE * IMG_SIZE * IMG_SIZE * sizeof(float), 
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_labels, batch_labels.data(), 
                       BATCH_SIZE * sizeof(int), 
                       cudaMemcpyHostToDevice);

            zero_grad(grads, BATCH_SIZE);

            // Forward pass
            forward(d_input, params, dropout_states, BATCH_SIZE, true);

            // Compute loss
            float loss;
            float *d_loss;
            cudaMalloc(&d_loss, sizeof(float));
            cudaMemset(d_loss, 0, sizeof(float));
            
            nll_loss<<<BATCH_SIZE, 1>>>(params.fc2_output, d_labels, d_loss, BATCH_SIZE, 10);
            
            cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
            total_loss += loss;

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA error before backward: %s\n", cudaGetErrorString(err));
        }

            backward(params, grads, d_labels, d_input);

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA error after backward: %s\n", cudaGetErrorString(err));
            }

            optimizer_step(params, grads, LEARNING_RATE);

            std::cout<<"Epoch: "<<epoch+1<< "\t "<< batch*BATCH_SIZE << " / "<< 60000<<" \t Batch loss: "<<loss <<std::endl;

            cudaFree(d_loss);
        }

        std::cout << "Epoch " << epoch + 1 << " Total loss: " << total_loss / num_batches << std::endl;
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_labels);
    freeNetwork(params);
    freeGradients(grads);
}

void test(std::vector<std::vector<float>> &test_images, std::vector<int> &test_labels) {
    NetworkParams params;
    DropoutStates dropout_states = {};
    
    // Initialize network 
    initializeNetwork(params, BATCH_SIZE);
    

    int num_batches = test_images.size() / BATCH_SIZE;
    int correct = 0;
    int total = 0;

    // Allocate device memory for input batch and labels
    float *d_input;
    int *d_labels;
    cudaMalloc(&d_input, BATCH_SIZE * IMG_SIZE * IMG_SIZE * sizeof(float));
    cudaMalloc(&d_labels, BATCH_SIZE * sizeof(int));

    std::vector<DetailedResult> detailed_results;

    for (int batch = 0; batch < num_batches; ++batch) {
        // Copy batch data to device
        std::vector<float> batch_images(BATCH_SIZE * IMG_SIZE * IMG_SIZE);
        std::vector<int> batch_labels(BATCH_SIZE);
        
        for (int i = 0; i < BATCH_SIZE; ++i) {
            std::copy(test_images[batch * BATCH_SIZE + i].begin(), 
                      test_images[batch * BATCH_SIZE + i].end(), 
                      batch_images.begin() + i * IMG_SIZE * IMG_SIZE);
            batch_labels[i] = test_labels[batch * BATCH_SIZE + i];
        }
        
        cudaMemcpy(d_input, batch_images.data(), 
                   BATCH_SIZE * IMG_SIZE * IMG_SIZE * sizeof(float), 
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_labels, batch_labels.data(), 
                   BATCH_SIZE * sizeof(int), 
                   cudaMemcpyHostToDevice);

        // Forward pass
        forward(d_input, params, dropout_states, BATCH_SIZE, false);

        // Copy predictions back to host
        std::vector<float> predictions(BATCH_SIZE * 10);
        cudaMemcpy(predictions.data(), params.fc2_output, 
                   BATCH_SIZE * 10 * sizeof(float), 
                   cudaMemcpyDeviceToHost);

        // Check predictions
        for (int i = 0; i < BATCH_SIZE; ++i) {
            // Manual implementation of finding max element
            int pred_label = 0;
            float max_prob = predictions[i * 10];
            for (int j = 1; j < 10; ++j) {
                if (predictions[i * 10 + j] > max_prob) {
                    max_prob = predictions[i * 10 + j];
                    pred_label = j;
                }
            }
            
            if (pred_label == batch_labels[i]) {
                correct++;
            }

            if (detailed_results.size() < 20 && 
                (pred_label != batch_labels[i] || detailed_results.size() < 10)) {
                DetailedResult result;
                result.true_label = batch_labels[i];
                result.pred_label = pred_label;
                result.probabilities = std::vector<float>(
                    predictions.begin() + i * 10, 
                    predictions.begin() + (i + 1) * 10
                );
                detailed_results.push_back(result);
            }
            
            total++;
        }
    }

    float accuracy = 100.0f * correct / total;
    std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_labels);
    freeNetwork(params);
}

int main() {
    std::vector<std::vector<float>> train_images, test_images;
    std::vector<int> train_labels, test_labels;

    // Read MNIST dataset
    read_mnist_images("MNIST_ORG/train-images.idx3-ubyte", train_images);
    read_mnist_labels("MNIST_ORG/train-labels.idx1-ubyte", train_labels);
    read_mnist_images("MNIST_ORG/t10k-images.idx3-ubyte", test_images);
    read_mnist_labels("MNIST_ORG/t10k-labels.idx1-ubyte", test_labels);

    std::cout << "Loaded MNIST: " << train_images.size() << " training, " << test_images.size() << " testing" << std::endl;

    // Training
    train(train_images, train_labels);

    // Testing
    test(test_images, test_labels);

    return 0;
}