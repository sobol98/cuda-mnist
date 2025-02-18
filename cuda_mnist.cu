/*
*
*  nvcc -o cuda_mnist cuda_mnist.cu -lcurand
*  ./cuda_mnist
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

struct DetailedResult {
    int true_label;
    int pred_label;
    std::vector<float> probabilities;
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


// RANDOM WEIGHTS INITIALIZATION
void initialize_weights(NetworkParams &params) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 42);  

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


void forward(float* input, NetworkParams &params, int batch_size, bool is_training) {
    dim3 blockSize(16, 16);
    
    // Conv1 + ReLU
    dim3 conv1_grid((IMG_SIZE + blockSize.x - 1) / blockSize.x, (IMG_SIZE + blockSize.y - 1) / blockSize.y, batch_size);
    conv2d<<<conv1_grid, blockSize>>>(input, params.conv1_output, params.conv1_weights, params.conv1_bias, batch_size, 1, 32, IMG_SIZE, 3);

    // Conv2 + ReLU
    dim3 conv2_grid((26 + blockSize.x - 1) / blockSize.x, (26 + blockSize.y - 1) / blockSize.y, batch_size);
    conv2d<<<conv2_grid, blockSize>>>(params.conv1_output, params.conv2_output, params.conv2_weights, params.conv2_bias, batch_size, 32, 64, 26, 3);

    // MaxPool
    dim3 pool_grid((24 + blockSize.x - 1) / blockSize.x, (24 + blockSize.y - 1) / blockSize.y, batch_size);
    maxpool2d<<<pool_grid, blockSize>>>(params.conv2_output, params.pool_output, batch_size, 64, 24, 2);

    // FC1 + ReLU
    dim3 fc1_grid((128 + blockSize.x - 1) / blockSize.x, 1, batch_size);
    linear<<<fc1_grid, blockSize>>>(params.pool_output, params.fc1_output, params.fc1_weights, params.fc1_bias, batch_size, 9216, 128);

    // FC2
    dim3 fc2_grid((10 + blockSize.x - 1) / blockSize.x, 1, batch_size);
    linear<<<fc2_grid, blockSize>>>(params.fc1_output, params.fc2_output, params.fc2_weights, params.fc2_bias, batch_size, 128, 10);

    // Log_softmax
    dim3 softmax_grid(batch_size);
    log_softmax<<<softmax_grid, 1>>>(params.fc2_output, params.fc2_output, batch_size, 10);
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


void test(std::vector<std::vector<float>> &test_images, std::vector<int> &test_labels) {
    NetworkParams params;
    
    initializeNetwork(params, BATCH_SIZE);
    initialize_weights(params);

    int num_batches = test_images.size() / BATCH_SIZE;
    int correct = 0;
    int total = 0;


    float *d_input;
    int *d_labels;
    cudaMalloc(&d_input, BATCH_SIZE * IMG_SIZE * IMG_SIZE * sizeof(float));
    cudaMalloc(&d_labels, BATCH_SIZE * sizeof(int));

    std::vector<DetailedResult> detailed_results;

    for (int batch = 0; batch < num_batches; ++batch) {
        std::vector<float> batch_images(BATCH_SIZE * IMG_SIZE * IMG_SIZE);
        std::vector<int> batch_labels(BATCH_SIZE);
        
        for (int i = 0; i < BATCH_SIZE; ++i) {
            std::copy(test_images[batch * BATCH_SIZE + i].begin(), test_images[batch * BATCH_SIZE + i].end(), batch_images.begin() + i * IMG_SIZE * IMG_SIZE);
            batch_labels[i] = test_labels[batch * BATCH_SIZE + i];
        }
        
        cudaMemcpy(d_input, batch_images.data(), BATCH_SIZE * IMG_SIZE * IMG_SIZE * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_labels, batch_labels.data(), BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice);

        forward(d_input, params, BATCH_SIZE, false);

        std::vector<float> predictions(BATCH_SIZE * 10);
        cudaMemcpy(predictions.data(), params.fc2_output, BATCH_SIZE * 10 * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < BATCH_SIZE; ++i) {
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

            if (detailed_results.size() < 20 && (pred_label != batch_labels[i] || detailed_results.size() < 10)) {
                DetailedResult result;
                result.true_label = batch_labels[i];
                result.pred_label = pred_label;
                result.probabilities = std::vector<float>( predictions.begin() + i * 10,  predictions.begin() + (i + 1) * 10);
                detailed_results.push_back(result);
                std::cout << "True: " << result.true_label << ", Pred: " << result.pred_label << std::endl;
            }

            total++;
        }
    }

    float accuracy = 100.0f * correct / total;
    std::cout << "Test Accuracy: " << accuracy << "%" << std::endl;

    cudaFree(d_input);
    cudaFree(d_labels);
    freeNetwork(params);
}


int main() {
    std::vector<std::vector<float>> train_images, test_images;
    std::vector<int> train_labels, test_labels;

    read_mnist_images("MNIST_ORG/train-images.idx3-ubyte", train_images);
    read_mnist_labels("MNIST_ORG/train-labels.idx1-ubyte", train_labels);

    read_mnist_images("MNIST_ORG/t10k-images.idx3-ubyte", test_images);
    read_mnist_labels("MNIST_ORG/t10k-labels.idx1-ubyte", test_labels);

    std::cout << "Loaded MNIST: " << train_images.size() << " training, " << test_images.size() << " testing" << std::endl;

    // eval
    test(test_images, test_labels);


    return 0;
}

