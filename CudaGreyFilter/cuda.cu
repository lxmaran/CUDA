
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include "cuda.h"

static const int THREADS = 28;
static const int SLICES = 1;
__global__ void grey_scale(char* imageData, int rows, int columns, int channels) {
	int thread_id = threadIdx.x;
	int from = rows/THREADS * thread_id;
	int to = rows/THREADS * (thread_id + 1);

	for (int x = from; x < to; x++) {
		for (int y = 0; y < columns; y++) {
			auto rgb = imageData[x * columns * channels + y * channels] * 0.3f;
			rgb += imageData[x * columns * channels + y * channels + 1] * 0.59f;
			rgb += imageData[x * columns * channels + y * channels + 2] * 0.11f;
			imageData[x * columns * channels + y * channels] = (char)(rgb);
			imageData[x * columns * channels + y * channels + 1] = (char)(imageData[x * columns * channels + y * channels]);
			imageData[x * columns * channels + y * channels + 2] = (char)(imageData[x * columns * channels + y * channels]);
		}
	}
}

void image_cuda(char *imageData, size_t size, int rows, int cols, int channels) {
	char *dev_image;

	cudaMalloc((void **)&dev_image, size);
	cudaMemcpy(dev_image, imageData, size, cudaMemcpyHostToDevice);

	grey_scale << < SLICES, THREADS >> > (dev_image, rows, cols, channels);

	cudaMemcpy(imageData, dev_image, size, cudaMemcpyDeviceToHost);
	cudaFree(dev_image);
}