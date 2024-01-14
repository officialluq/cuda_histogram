#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

const int numBins = 256;

__global__ void calculateHistogram(const unsigned char* inputImage, int* histogramR, int* histogramG, int* histogramB, int imageSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < imageSize) {
        atomicAdd(&histogramR[inputImage[3 * tid]], 1);
        atomicAdd(&histogramG[inputImage[3 * tid + 1]], 1);
        atomicAdd(&histogramB[inputImage[3 * tid + 2]], 1);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_file_path>" << std::endl;
        return -1;
    }

    const char* imagePath = argv[1];

    std::ifstream file(imagePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the image file." << std::endl;
        return -1;
    }

    file.seekg(0, std::ios::end);
    int imageSize = file.tellg();
    file.seekg(0, std::ios::beg);

    unsigned char* h_image = new unsigned char[imageSize];
    file.read(reinterpret_cast<char*>(h_image), imageSize);
    file.close();

    unsigned char* d_image;
    cudaMalloc(&d_image, imageSize * sizeof(unsigned char));
    cudaMemcpy(d_image, h_image, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);

    int* d_histogramR;
    int* d_histogramG;
    int* d_histogramB;
    cudaMalloc(&d_histogramR, numBins * sizeof(int));
    cudaMalloc(&d_histogramG, numBins * sizeof(int));
    cudaMalloc(&d_histogramB, numBins * sizeof(int));
    cudaMemset(d_histogramR, 0, numBins * sizeof(int));
    cudaMemset(d_histogramG, 0, numBins * sizeof(int));
    cudaMemset(d_histogramB, 0, numBins * sizeof(int));

    int blockSize = 32;
    int gridSize = (imageSize/3 + blockSize - 1) / blockSize;
    std::cout << "Size: " << gridSize << std::endl;

    // Run kernel
    calculateHistogram<<<gridSize, blockSize>>>(d_image, d_histogramR, d_histogramG, d_histogramB, imageSize);

    int h_histogramR[numBins];
    int h_histogramG[numBins];
    int h_histogramB[numBins];
    cudaMemcpy(h_histogramR, d_histogramR, numBins * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_histogramG, d_histogramG, numBins * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_histogramB, d_histogramB, numBins * sizeof(int), cudaMemcpyDeviceToHost);

    std::ofstream csvFileR("histogram_red.csv");
    std::ofstream csvFileG("histogram_green.csv");
    std::ofstream csvFileB("histogram_blue.csv");
    if (csvFileR.is_open() && csvFileG.is_open() && csvFileB.is_open()) {
        csvFileR << "Bin,Count" << std::endl;
        csvFileG << "Bin,Count" << std::endl;
        csvFileB << "Bin,Count" << std::endl;

        for (int i = 0; i < numBins; ++i) {
            csvFileR << i << "," << h_histogramR[i] << std::endl;
            csvFileG << i << "," << h_histogramG[i] << std::endl;
            csvFileB << i << "," << h_histogramB[i] << std::endl;
        }

        csvFileR.close();
        csvFileG.close();
        csvFileB.close();

        std::cout << "Histogram data saved to histogram_red.csv, histogram_green.csv, and histogram_blue.csv" << std::endl;
    } else {
        std::cerr << "Error: Could not open the CSV files for writing." << std::endl;
    }

    // Free alloc mem
    delete[] h_image;
    cudaFree(d_image);
   
}