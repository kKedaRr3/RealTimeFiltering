#include "CudaFilters.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 1. DEFINICJA KERNELA (Musi być u góry!)
__global__ void thresholdKernel(unsigned char* data, int numPixels, unsigned char threshold) {
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (pixelIdx < numPixels) {
        int offset = pixelIdx * 3; // BGR format (3 bajty na piksel)

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
//  bo rozmiar tablicy shared musi byc znany przy kompilacji 
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
        currentBufferSize = 0;
    }
}

// 3. GŁÓWNA FUNKCJA WYWOŁYWANA Z C++
void applyThresholdCuda(unsigned char* data, int width, int height, int channels, unsigned char threshold) {
    // Jeśli z jakiegoś powodu bufor nie został stworzony, nie rób nic
    if (d_buffer == nullptr || data == nullptr || width <= 0 || height <= 0) {
        return;
    }

    // Upewnij się, że obsługujemy tylko 3 kanały, bo kernel ma na sztywno "offset * 3"
    if (channels != 3) return;

    int numPixels = width * height;
    int totalBytes = numPixels * channels;

    cudaMemcpy(d_buffer, data, totalBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
    thresholdKernel << <blocksPerGrid, threadsPerBlock >> > (d_buffer, numPixels, threshold);

    cudaMemcpy(data, d_buffer, totalBytes, cudaMemcpyDeviceToHost);
}

// nie testowalem ale na colabie dziala
void applyLowPassCuda(unsigned char* data, int width, int height){
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
    filter3x3_LowPass<<<dimGrid, dimBlock>>>(d_buffer, height, width );

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