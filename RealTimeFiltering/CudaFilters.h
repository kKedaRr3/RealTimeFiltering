#pragma once
bool isCudaAvailable(); 
// Funkcja wywoływana raz na początku
void initCudaBuffer(int width, int height, int channels);

// Funkcja wywoływana co klatkę (już bez malloc!)
void applyThresholdCuda(unsigned char* data, int width, int height, int channels, unsigned char threshold);

void applyLowPassCuda(unsigned char* data, int width, int height);

void applyHighPassCuda(unsigned char* data, int width, int height);

// Funkcja wywoływana na końcu
void freeCudaBuffer();