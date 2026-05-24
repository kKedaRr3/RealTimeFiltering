#include "CudaFilters.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 1. Thresholding kernel
__global__ void thresholdKernel(unsigned char* data, int numPixels, unsigned char threshold) {
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pixelIdx < numPixels) {
        int offset = pixelIdx * 3;

        // Odczytujemy składowe (OpenCV standardowo używa BGR)
        unsigned char b = data[offset];
        unsigned char g = data[offset + 1];
        unsigned char r = data[offset + 2];

        // Liczymy jasność
        unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);

        // Binaryzacja
        unsigned char res = (gray > threshold) ? 255 : 0;

        // Zapisujemy wynik do wszystkich 3 kanałów
        data[offset] = res;
        data[offset + 1] = res;
        data[offset + 2] = res;
    }
}

// TILE_WIDTH musi byc znane przy kompilacji 
// bo rozmiar tablicy shared musi byc znany przy kompilacji 
#define TILE_WIDTH 16

// 3x3 filter kernel with average low pass filter
__global__ void filter3x3_LowPass(unsigned char* data, int sizeV, int sizeH)
{
    const int sharedSize = TILE_WIDTH + 2;

    // Shared memory for BGR image
    __shared__ unsigned char
        tile[sharedSize][sharedSize][3];

    // Local coordinates
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Flattened thread id
    int tid = ty * TILE_WIDTH + tx;

    // Global pixel coordinates
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // Load values into shared memory
    for (int index = tid;
        index < sharedSize * sharedSize;
        index += TILE_WIDTH * TILE_WIDTH)
    {
        int sRow = index / sharedSize;
        int sCol = index % sharedSize;

        // Corresponding global coordinates
        int gRow =
            blockIdx.y * TILE_WIDTH + sRow - 1;

        int gCol =
            blockIdx.x * TILE_WIDTH + sCol - 1;

        if (gRow >= 0 && gRow < sizeV &&
            gCol >= 0 && gCol < sizeH)
        {
            int globalOffset =
                (gRow * sizeH + gCol) * 3;

            tile[sRow][sCol][0] =
                data[globalOffset + 0]; // B

            tile[sRow][sCol][1] =
                data[globalOffset + 1]; // G

            tile[sRow][sCol][2] =
                data[globalOffset + 2]; // R
        }
        else
        {
            tile[sRow][sCol][0] = 0;
            tile[sRow][sCol][1] = 0;
            tile[sRow][sCol][2] = 0;
        }
    }

    __syncthreads();

    // Apply 3x3 filter
    // Average
    if (row < sizeV && col < sizeH)
    {
        float sumB = 0.0f;
        float sumG = 0.0f;
        float sumR = 0.0f;

#pragma unroll
        for (int dy = -1; dy <= 1; dy++)
        {
#pragma unroll
            for (int dx = -1; dx <= 1; dx++)
            {
                int sy = ty + 1 + dy;
                int sx = tx + 1 + dx;

                sumB += tile[sy][sx][0];
                sumG += tile[sy][sx][1];
                sumR += tile[sy][sx][2];
            }
        }

        // average blur
        sumB /= 9.0f;
        sumG /= 9.0f;
        sumR /= 9.0f;

        // Output values
        int outputOffset = (row * sizeH + col) * 3;
        data[outputOffset + 0] = (unsigned char)sumB;
        data[outputOffset + 1] = (unsigned char)sumG;
        data[outputOffset + 2] = (unsigned char)sumR;
    }
}

__global__ void filter3x3_HighPass(unsigned char* data, int sizeV, int sizeH) {
    const int sharedSize = TILE_WIDTH + 2;
    __shared__ unsigned char tile[sharedSize][sharedSize][3];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * TILE_WIDTH + tx;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // Ładowanie do pamięci współdzielonej 
    for (int index = tid; index < sharedSize * sharedSize; index += TILE_WIDTH * TILE_WIDTH) {
        int sRow = index / sharedSize;
        int sCol = index % sharedSize;
        int gRow = blockIdx.y * TILE_WIDTH + sRow - 1;
        int gCol = blockIdx.x * TILE_WIDTH + sCol - 1;

        if (gRow >= 0 && gRow < sizeV && gCol >= 0 && gCol < sizeH) {
            int globalOffset = (gRow * sizeH + gCol) * 3;
            tile[sRow][sCol][0] = data[globalOffset + 0];
            tile[sRow][sCol][1] = data[globalOffset + 1];
            tile[sRow][sCol][2] = data[globalOffset + 2];
        }
        else {
            tile[sRow][sCol][0] = tile[sRow][sCol][1] = tile[sRow][sCol][2] = 0;
        }
    }

    __syncthreads();

    if (row < sizeV && col < sizeH) {
        float resB = 0.0f, resG = 0.0f, resR = 0.0f;

        float kernel[3][3] = {
            {-0.5f, -0.5f, -0.5f},
            {-0.5f,  5.0f, -0.5f},
            {-0.5f, -0.5f, -0.5f}
        };

        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                float weight = kernel[dy + 1][dx + 1];
                resB += (float)tile[ty + 1 + dy][tx + 1 + dx][0] * weight;
                resG += (float)tile[ty + 1 + dy][tx + 1 + dx][1] * weight;
                resR += (float)tile[ty + 1 + dy][tx + 1 + dx][2] * weight;
            }
        }

        int outputOffset = (row * sizeH + col) * 3;
        // Używamy fminf/fmaxf do przycięcia wyników do zakresu 0-255
        data[outputOffset + 0] = (unsigned char)fmaxf(0.0f, fminf(255.0f, resB));
        data[outputOffset + 1] = (unsigned char)fmaxf(0.0f, fminf(255.0f, resG));
        data[outputOffset + 2] = (unsigned char)fmaxf(0.0f, fminf(255.0f, resR));
    }
}

// Sobel edge detection
__global__ void sobelKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= height || col >= width) return;

    int outOffset = (row * width + col) * 3;

    // Brzegi ustawiamy na czarno, bo tam nie ma pełnego sąsiedztwa 3x3
    if (row == 0 || col == 0 || row == height - 1 || col == width - 1) {
        output[outOffset + 0] = 0;
        output[outOffset + 1] = 0;
        output[outOffset + 2] = 0;
        return;
    }

    int gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    float sumX = 0.0f;
    float sumY = 0.0f;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int offset = ((row + dy) * width + (col + dx)) * 3;

            unsigned char b = input[offset + 0];
            unsigned char g = input[offset + 1];
            unsigned char r = input[offset + 2];

            float gray = 0.299f * r + 0.587f * g + 0.114f * b;

            sumX += gray * gx[dy + 1][dx + 1];
            sumY += gray * gy[dy + 1][dx + 1];
        }
    }

    float value = sqrtf(sumX * sumX + sumY * sumY);

    value = fmaxf(0.0f, fminf(255.0f, value));

    unsigned char edge = (unsigned char)value;

    output[outOffset + 0] = edge;
    output[outOffset + 1] = edge;
    output[outOffset + 2] = edge;
}

// 2. ZMIENNE I FUNKCJE POMOCNICZE
unsigned char* d_buffer = nullptr;
unsigned char* d_output = nullptr;
int currentBufferSize = 0;

void initCudaBuffer(int width, int height, int channels) {
    int size = width * height * channels;

    if (d_buffer == nullptr || size != currentBufferSize) {
        if (d_buffer != nullptr) cudaFree(d_buffer);
        if (d_output != nullptr) cudaFree(d_output);

        cudaMalloc(&d_buffer, size);
        cudaMalloc(&d_output, size);

        currentBufferSize = size;
    }
}

void freeCudaBuffer() {
    if (d_buffer != nullptr) {
        cudaFree(d_buffer);
        d_buffer = nullptr;
    }

    if (d_output != nullptr) {
        cudaFree(d_output);
        d_output = nullptr;
    }

    currentBufferSize = 0;
}

void applyThresholdCuda(unsigned char* data, int width, int height, int channels, unsigned char threshold) {
    if (d_buffer == nullptr || data == nullptr || width <= 0 || height <= 0) {
        return;
    }

    if (channels != 3) return;

    int numPixels = width * height;
    int totalBytes = numPixels * channels;

    cudaMemcpy(d_buffer, data, totalBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
    thresholdKernel << <blocksPerGrid, threadsPerBlock >> > (d_buffer, numPixels, threshold);

    cudaMemcpy(data, d_buffer, totalBytes, cudaMemcpyDeviceToHost);
}

void applyLowPassCuda(unsigned char* data, int width, int height) {
    if (d_buffer == nullptr || data == nullptr || width <= 0 || height <= 0) {
        return;
    }

    int numPixels = width * height;
    int totalBytes = numPixels * 3 * sizeof(unsigned char);

    cudaMemcpy(d_buffer, data, totalBytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(
        (width - 1) / TILE_WIDTH + 1,
        (height - 1) / TILE_WIDTH + 1
    );

    filter3x3_LowPass << <dimGrid, dimBlock >> > (d_buffer, height, width);

    cudaMemcpy(data, d_buffer, totalBytes, cudaMemcpyDeviceToHost);
}

void applyHighPassCuda(unsigned char* data, int width, int height) {
    if (d_buffer == nullptr || data == nullptr || width <= 0 || height <= 0) {
        return;
    }

    int size = width * height * 3;

    cudaMemcpy(d_buffer, data, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(
        (width - 1) / TILE_WIDTH + 1,
        (height - 1) / TILE_WIDTH + 1
    );

    filter3x3_HighPass << <dimGrid, dimBlock >> > (d_buffer, height, width);

    cudaMemcpy(data, d_buffer, size, cudaMemcpyDeviceToHost);
}

void applyEdgeDetectionCuda(unsigned char* data, int width, int height) {
    if (d_buffer == nullptr || d_output == nullptr || data == nullptr || width <= 0 || height <= 0) {
        return;
    }

    int totalBytes = width * height * 3 * sizeof(unsigned char);

    cudaMemcpy(d_buffer, data, totalBytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(
        (width - 1) / TILE_WIDTH + 1,
        (height - 1) / TILE_WIDTH + 1
    );

    sobelKernel << <dimGrid, dimBlock >> > (d_buffer, d_output, width, height);

    cudaMemcpy(data, d_output, totalBytes, cudaMemcpyDeviceToHost);
}

bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        return false;
    }

    return true;
}