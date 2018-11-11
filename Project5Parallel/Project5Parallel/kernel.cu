/******************************
*STUDENT NAME: DAVID PARKS    *
*PROJECT: 6 - GREY SCALE FLIP *
*DUE DATE: THURS 18/10/18     *
*******************************/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include<stdio.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>

#define PPM_MAGIC_1 'P'
#define PPM_MAGIC_2 '6' 
#define BLOCK_SIZE 16;

struct PPM_header {
	int width;
	int height;
	int max_color;
};
struct RGB_8 {
	uint8_t r;
	uint8_t g;
	uint8_t b;
};//__attribute__((packed));

void PPM_read_header(std::ifstream &inp, PPM_header &ppm_header) {
	char ppm_magic_1, ppm_magic_2;
	inp >> ppm_magic_1;
	inp >> ppm_magic_2;

	if (ppm_magic_1 != PPM_MAGIC_1 || ppm_magic_2 != PPM_MAGIC_2) {
		throw std::runtime_error("File does not begin with PPM magic number");
	}

	int width;
	inp >> width;
	ppm_header.width = width;
	int height;
	inp >> height;
	ppm_header.height = height;

	int max_color;
	inp >> max_color;
	ppm_header.max_color = max_color;

	char space;
	//inp >> space;		// finish the header
	inp.read(&space, 1);

	return;
}

void PPM_read_rgb_8(std::ifstream &inp, int width, int height, RGB_8 *img) {
	inp.read((char *)img, sizeof(RGB_8)*width*height);
	if (!inp) {
		std::stringstream ss;
		ss << "error: only " << inp.gcount() << " could be read";
		throw std::runtime_error(ss.str());
	}
}

void PPM_write_header_8(std::ofstream &outp, int width, int height) {
	// write the header
	outp << PPM_MAGIC_1 << PPM_MAGIC_2 << (char)10 << width << (char)10
		<< height << (char)10 << 255 << (char)10;
}

void PPM_write_rgb_8(std::ofstream &outp, int width, int height, RGB_8 *img) {
	// write the image
	outp.write((char *)img, sizeof(RGB_8) * width * height);
	if (!outp) {
		std::stringstream ss;
		ss << "error: only " << outp.tellp() << " could be written";
		throw std::runtime_error(ss.str());
	}
}

////////////////////// STUDENT CODE (1/2) //////////////////////


__global__ void gray_scale_flip(RGB_8* img, int height, int width)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	if (row < height && col < width / 2)
	{
		int i = row * width + col;
		//temp var for slip pixel
		RGB_8 temp = img[(row + 1) * width - col - 1];

		//computing gray value
		float gray_value = 0.21 * img[i].r + 0.72 * img[i].g + 0.07 * img[i].b;
		img[i].r = gray_value;
		img[i].g = gray_value;
		img[i].b = gray_value;

		//set flip pixel to grayed current pixel
		img[(row + 1) * width - col - 1] = img[i];

		//set current pixel to temp pixel
		img[i] = temp;

		//computing gray value
		gray_value = 0.21 * img[i].r + 0.72 * img[i].g + 0.07 * img[i].b;
		img[i].r = gray_value;
		img[i].g = gray_value;
		img[i].b = gray_value;
	}
}

////////////////////// END STUDENT CODE (1/2) //////////////////////


int main(int argc, char *argv[]) {
	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " in_ppm_file out_ppm_file" << std::endl;
		return 1;
	}

	PPM_header img_header;
	cudaError_t cudaStatus;


	try {
		std::ifstream ifs(argv[1], std::ios::binary);
		if (!ifs) {
			throw std::runtime_error("Cannot open input file");
		}

		PPM_read_header(ifs, img_header);
		std::cout << img_header.width << " " << img_header.height << " " << img_header.max_color << std::endl;

		RGB_8 *img = new RGB_8[img_header.height * img_header.width];

		PPM_read_rgb_8(ifs, img_header.width, img_header.height, (RGB_8 *)img);

		std::ofstream ofs(argv[2], std::ios::binary);
		if (!ofs) {
			throw std::runtime_error("Cannot open output file");
		}

		PPM_write_header_8(ofs, img_header.width, img_header.height);


		////////////////////// STUDENT CODE (2/2) //////////////////////
		RGB_8* img_device;
		int size = img_header.height * img_header.width;

		//allocate memory on GPU
		cudaStatus = cudaMalloc((void**)&img_device, size * sizeof(RGB_8));
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Could not allocate space on GPU" << std::endl;
			cudaFree(img_device);
			return -1;
		}

		//copy memory from host to GPU
		cudaStatus = cudaMemcpy(img_device, img, size * sizeof(RGB_8), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Could not copy memory from host to GPU" << std::endl;
			cudaFree(img_device);
			return -2;
		}

		//define block
		dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE, 1);

		//define grid
		dim3 dim_grid(ceil(((float)img_header.width / 2) / BLOCK_SIZE), ceil((float)img_header.height / BLOCK_SIZE), 1);

		//call kernel function
		gray_scale_flip << <dim_grid, dim_block >> >(img_device, img_header.height, img_header.width);

		RGB_8 *new_img = (RGB_8*)malloc(sizeof(RGB_8) * img_header.height * img_header.width);

		//copy memory from host to GPU
		cudaStatus = cudaMemcpy(new_img, img_device, size * sizeof(RGB_8), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess)
		{
			std::cout << "Could not copy memory from GPU to host" << std::endl;
			cudaFree(img_device);
			return -3;
		}

		////////////////////// END STUDENT CODE (2/2) //////////////////////


		PPM_write_rgb_8(ofs, img_header.width, img_header.height, (RGB_8 *)new_img);

		cudaFree(img_device);

		ifs.close();
		ofs.close();
	}
	catch (std::runtime_error &re) {
		std::cout << re.what() << std::endl;
		return 2;
	}

	return 0;
}
