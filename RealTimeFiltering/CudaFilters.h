#pragma once

bool isCudaAvailable();

// Funkcja wywoływana raz na początku
void initCudaBuffer(int width, int height, int channels);

// Funkcje wywoływane co klatkę
void applyThresholdCuda(unsigned char* data, int width, int height, int channels, unsigned char threshold);

void applyLowPassCuda(unsigned char* data, int width, int height);

void applyHighPassCuda(unsigned char* data, int width, int height);

void applyEdgeDetectionCuda(unsigned char* data, int width, int height);

// Funkcja wywoływana na końcu
void freeCudaBuffer();