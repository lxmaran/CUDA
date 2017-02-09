//
// Created by robert on 1/17/17.
//
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "cuda.h"
using namespace std;
static const char *const IMAGE = "tux-288.png";

using namespace std::chrono;

int main() {
	milliseconds chrono = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	cv::Mat image = cv::imread(IMAGE, 1);
	if (!image.data) {
		std::cout << "Not found\n";
		return -1;
	}

	uint8_t *imageData = image.data;

	if (image.isContinuous()) {
		image_cuda((char *)imageData, (image.total() * image.elemSize()), image.rows, image.cols, image.channels());
	}
	else {
		std::cout << "inconsistent img\n";
	}

	cv::imwrite("laba.png", image);
	milliseconds end = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
	std::cout << (end - chrono).count() << endl;
}