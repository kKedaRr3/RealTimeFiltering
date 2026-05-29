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

__global__ void threshold(unsigned char* data, int height, int width, unsigned char threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int pixelIdx = y * width + x;
    int offset = pixelIdx * 3;

    unsigned char b = data[offset];
    unsigned char g = data[offset + 1];
    unsigned char r = data[offset + 2];

    unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    unsigned char res = (gray > threshold) ? 255 : 0;

    data[offset] = res;
    data[offset + 1] = res;
    data[offset + 2] = res;
}

// TILE_WIDTH musi byc znane przy kompilacji 
// bo rozmiar tablicy shared musi byc znany przy kompilacji 
#define TILE_WIDTH 16

// Filter array in GPU constant memory
__constant__ float c_filter[3][3];

// generic 3x3 filter kernel
__global__ void filter3x3(unsigned char* data, int sizeV, int sizeH, float mix) {
    if(mix == 0.0f) return;

    const int sharedSize = TILE_WIDTH + 2;
    __shared__ unsigned char
        tile[sharedSize][sharedSize][3];

    // Flattened thread id
    int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;

    // Global pixel coordinates
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Load values into shared memory
    for (int index = tid; 
            index < sharedSize * sharedSize; 
            index += TILE_WIDTH * TILE_WIDTH) {
        int sRow = index / sharedSize;
        int sCol = index % sharedSize;
        int gRow = blockIdx.y * TILE_WIDTH + sRow - 1;
        int gCol = blockIdx.x * TILE_WIDTH + sCol - 1;

        if (gRow >= 0 && gRow < sizeV && gCol >= 0 && gCol < sizeH) {
            int globalOffset = (gRow * sizeH + gCol) * 3;
            tile[sRow][sCol][0] = data[globalOffset + 0]; // B
            tile[sRow][sCol][1] = data[globalOffset + 1]; // G
            tile[sRow][sCol][2] = data[globalOffset + 2]; // R
        } else {
            tile[sRow][sCol][0] = 0;
            tile[sRow][sCol][1] = 0;
            tile[sRow][sCol][2] = 0;
        }
    }

    __syncthreads();

    // Apply 3x3 filter
    if (row < sizeV && col < sizeH)
    {
        float sumB = 0.0f;
        float sumG = 0.0f;
        float sumR = 0.0f;

        #pragma unroll
        for (int dy=0; dy<=2; ++dy) {
            #pragma unroll
            for (int dx=0; dx<=2; ++dx) {
                int sy = threadIdx.y + dy;
                int sx = threadIdx.x + dx;
                sumB += c_filter[dy][dx] * tile[sy][sx][0];
                sumG += c_filter[dy][dx] * tile[sy][sx][1];
                sumR += c_filter[dy][dx] * tile[sy][sx][2];
            }
        }

        // Output clamped and weighed
        int outputOffset = (row * sizeH + col) * 3;
        data[outputOffset + 0] = (unsigned char)(
            tile[ty + 1][tx + 1][0] * (1.0f - mix) +
            fminf(255.0f, fmaxf(0.0f, sumB)) * mix
        );
        data[outputOffset + 1] = (unsigned char)(
            tile[ty + 1][tx + 1][1] * (1.0f - mix) +
            fminf(255.0f, fmaxf(0.0f, sumG)) * mix
        );
        data[outputOffset + 2] = (unsigned char)(
            tile[ty + 1][tx + 1][2] * (1.0f - mix) +
            fminf(255.0f, fmaxf(0.0f, sumR)) * mix
        );
    }
}

// obsolete
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

// obsolete
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

//Sobel edge detection
__global__ void sobelKernel(unsigned char* input, unsigned char* output, int width, int height)
{
    const int sharedSize = TILE_WIDTH + 2;
    __shared__ unsigned char tile[sharedSize][sharedSize];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + tx;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int tid = ty * TILE_WIDTH + tx;

    //wczytanie kafelka 16x16 + obwódki 1 piksel do shared memory
    for (int i = tid; i < sharedSize * sharedSize; i += TILE_WIDTH * TILE_WIDTH)
    {
        int sRow = i / sharedSize;
        int sCol = i % sharedSize;

        int gRow = blockIdx.y * TILE_WIDTH + sRow - 1;
        int gCol = blockIdx.x * TILE_WIDTH + sCol - 1;

        if (gRow >= 0 && gRow < height && gCol >= 0 && gCol < width)
        {
            int offset = (gRow * width + gCol) * 3;

            unsigned char b = input[offset + 0];
            unsigned char g = input[offset + 1];
            unsigned char r = input[offset + 2];

            unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);

            tile[sRow][sCol] = gray;
        }
        else
        {
            tile[sRow][sCol] = 0;
        }
    }

    __syncthreads();

    if (row >= height || col >= width)
        return;

    int outOffset = (row * width + col) * 3;

    // Brzegi obrazu ustawiamy na czarno
    if (row == 0 || col == 0 || row == height - 1 || col == width - 1)
    {
        output[outOffset + 0] = 0;
        output[outOffset + 1] = 0;
        output[outOffset + 2] = 0;
        return;
    }

    int gx = -tile[ty][tx] + tile[ty][tx + 2] - 2 * tile[ty + 1][tx] + 2 * tile[ty + 1][tx + 2] - tile[ty + 2][tx] + tile[ty + 2][tx + 2];
    int gy = -tile[ty][tx] - 2 * tile[ty][tx + 1] - tile[ty][tx + 2] + tile[ty + 2][tx] + 2 * tile[ty + 2][tx + 1] + tile[ty + 2][tx + 2];

    unsigned char edge = (unsigned char)fminf(255.0f,sqrtf((float)(gx * gx + gy * gy))
    );

    output[outOffset + 0] = edge;
    output[outOffset + 1] = edge;
    output[outOffset + 2] = edge;
}

__global__ void median3x3(unsigned char* data, int sizeV, int sizeH, float mix) {
    if(mix == 0.0f) return;

    const int sharedSize = TILE_WIDTH + 2;
    __shared__ unsigned char
        tile[sharedSize][sharedSize][3];

    // Flattened thread id
    int tid = threadIdx.y * TILE_WIDTH + threadIdx.x;

    // Global pixel coordinates
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    // Load values into shared memory
    for (int index = tid; 
            index < sharedSize * sharedSize; 
            index += TILE_WIDTH * TILE_WIDTH) {
        int sRow = index / sharedSize;
        int sCol = index % sharedSize;
        int gRow = blockIdx.y * TILE_WIDTH + sRow - 1;
        int gCol = blockIdx.x * TILE_WIDTH + sCol - 1;

        if (gRow >= 0 && gRow < sizeV && gCol >= 0 && gCol < sizeH) {
            int globalOffset = (gRow * sizeH + gCol) * 3;
            tile[sRow][sCol][0] = data[globalOffset + 0]; // B
            tile[sRow][sCol][1] = data[globalOffset + 1]; // G
            tile[sRow][sCol][2] = data[globalOffset + 2]; // R
        } else {
            tile[sRow][sCol][0] = 0;
            tile[sRow][sCol][1] = 0;
            tile[sRow][sCol][2] = 0;
        }
    }

    __syncthreads();

    // Apply 3x3 filter
    if (row < sizeV && col < sizeH)
    {
        unsigned char b[9];
        unsigned char g[9];
        unsigned char r[9];

        int k = 0;
        #pragma unroll
        for (int dy = 0; dy < 3; ++dy) {
            #pragma unroll
            for (int dx = 0; dx < 3; ++dx) {
                int sy = ty + dy;
                int sx = tx + dx;
                unsigned char vb = tile[sy][sx][0];
                unsigned char vg = tile[sy][sx][1];
                unsigned char vr = tile[sy][sx][2];

                int i = k - 1;
                while (i >= 0 && b[i] > vb) {
                    b[i + 1] = b[i];
                    i--;
                }
                b[i + 1] = vb;

                i = k - 1;
                while (i >= 0 && g[i] > vg) {
                    g[i + 1] = g[i];
                    i--;
                }
                g[i + 1] = vg;

                i = k - 1;
                while (i >= 0 && r[i] > vr) {
                    r[i + 1] = r[i];
                    i--;
                }
                r[i + 1] = vr;

                k++;
            }
        }

        if (row == 0 || col == 0 || row == height - 1 || col == width - 1)
            return;

        int outputOffset = (row * sizeH + col) * 3;
        data[outputOffset + 0] = b[4]*mix + tile[ty + 1][tx + 1][0]*(1.0f-mix);
        data[outputOffset + 1] = g[4]*mix + tile[ty + 1][tx + 1][1]*(1.0f-mix);
        data[outputOffset + 2] = r[4]*mix + tile[ty + 1][tx + 1][2]*(1.0f-mix);
    }
}

__global__ void posterize(unsigned char* data, int height, int width, float degree) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int pixelIdx = y * width + x;
    int offset = pixelIdx * 3;

    // Exponential perceptual mapping:
    // 0.0 -> 256 levels
    // 0.5 -> 16 levels
    // 1.0 -> 2 levels
    float exponent = 8.0f * (1.0f - degree);
    int levels = max(2, (int)roundf(powf(2.0f, exponent)));
    float step = 255.0f / (levels - 1);

    // loops over 3 channels
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float value = (float)data[offset + c];
        value = roundf(value / step) * step;
        data[offset + c] = (unsigned char)(
            fminf(255.0f, fmaxf(0.0f, value))
        );
    }
}



// 2. ZMIENNE I FUNKCJE POMOCNICZE
unsigned char* d_buffer = nullptr;
int currentBufferSize = 0;

void initCudaBuffer(int width, int height, int channels) {
    int size = width * height * channels;

    if (d_buffer == nullptr || size != currentBufferSize) {
        if (d_buffer != nullptr) cudaFree(d_buffer);
        cudaMalloc(&d_buffer, size);

        currentBufferSize = size;
    }
}

void freeCudaBuffer() {
    if (d_buffer != nullptr) {
        cudaFree(d_buffer);
        d_buffer = nullptr;
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

void applyFilterCuda(unsigned char* data, int width, int height, float filter[3][3], float mix) {
    if (d_buffer == nullptr || data == nullptr || width <= 0 || height <= 0)
        return;

    int numPixels = width * height;
    int totalBytes = numPixels * 3 * sizeof(unsigned char);

    cudaMemcpy(d_buffer, data, totalBytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_filter, filter, sizeof(float) * 9);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(
        (width - 1) / TILE_WIDTH + 1,
        (height - 1) / TILE_WIDTH + 1
    );

    filter3x3<<<dimGrid, dimBlock>>> (d_buffer, height, width, mix);

    cudaMemcpy(data, d_buffer, totalBytes, cudaMemcpyDeviceToHost);
}

// obsolete
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

// obsolete
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
    if (d_buffer == nullptr || data == nullptr || width <= 0 || height <= 0) {
        return;
    }

    int totalBytes = width * height * 3 * sizeof(unsigned char);
    unsigned char* d_output = nullptr;

    cudaMalloc(&d_output, totalBytes);
    cudaMemcpy(d_buffer, data, totalBytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH,(height + TILE_WIDTH - 1) / TILE_WIDTH);

    sobelKernel << <dimGrid, dimBlock >> > (d_buffer, d_output, width, height);
    cudaMemcpy(data, d_output, totalBytes, cudaMemcpyDeviceToHost);
    cudaFree(d_output);
}


void runCuda(unsigned char* data, int width, int height, float mixFactors[10]) {
// Pass mix factors for all possible effects as array of floats 0-1
// If an effect is not enabled, pass -1
// Effects:
// 0 - Threshold
// 1 - Low pass
// 2 - High pass
// 3 - Sobel
// 4 - Median*
// 5 - Posterize*
// 6 - 
// 7 - 
// 8 - 
// 9 - 

    if (d_buffer == nullptr || data == nullptr || width <= 0 || height <= 0)
        return;

    int numPixels = width * height;
    int totalBytes = numPixels * 3 * sizeof(unsigned char);

    cudaMemcpy(d_buffer, data, totalBytes, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(
        (width - 1) / TILE_WIDTH + 1,
        (height - 1) / TILE_WIDTH + 1
    );

    if(mixFactors[0] >= 0.0f)
        threshold<<<blocksPerGrid, threadsPerBlock>>>(d_buffer, height, width, (unsigned char)(255*mixFactors[0]));
    if(mixFactors[1] >= 0.0f) {
        cudaMemcpyToSymbol(c_filter, {
            {1.f/9, 1.f/9, 1.f/9},
            {1.f/9, 1.f/9, 1.f/9},
            {1.f/9, 1.f/9, 1.f/9}
        }, sizeof(float) * 9);
        filter3x3<<<dimGrid, dimBlock>>> (d_buffer, height, width, mixFactors[1]);
    }
    if(mixFactors[2] >= 0.0f) {
        cudaMemcpyToSymbol(c_filter, {
            { 0.f, -1.f,  0.f },
            {-1.f,  5.f, -1.f },
            { 0.f, -1.f,  0.f }
        }, sizeof(float) * 9);
        filter3x3<<<dimGrid, dimBlock>>> (d_buffer, height, width, mixFactors[2]);
    }
    if(mixFactors[3] >= 0.0f)
        SobelKernel<<<blocksPerGrid, threadsPerBlock>>>(d_buffer, width, height);
    if(mixFactors[4] >= 0.0f)
        median3x3<<<blocksPerGrid, threadsPerBlock>>>(d_buffer, height, width, mixFactors[4]);
    if(mixFactors[5] >= 0.0f)
        posterize<<<blocksPerGrid, threadsPerBlock>>>(d_buffer, height, width, mixFactors[5]);
    

    
    cudaMemcpy(data, d_buffer, totalBytes, cudaMemcpyDeviceToHost);
}

bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0) {
        return false;
    }

    return true;
}