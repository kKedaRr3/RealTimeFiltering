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

bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess || deviceCount == 0) {
        return false;
    }
    return true;
}