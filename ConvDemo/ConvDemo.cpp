// ConvDemo.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <math.h>
#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>

#ifndef MAX
#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef PI 
#define PI 3.14159265358979323846
#endif

#ifndef EPS
#define EPS 0.0000000000001
#endif

using namespace cv;
using namespace std;

// Memory allocation for output should be done by caller
template<typename DType>
void im2col(DType *data, int data_h, int data_w,
	int *kernel, int kernel_h, int kernel_w,
	DType *output)
{
	int pad_h = kernel_h / 2;
	int pad_w = kernel_w / 2;

	int output_h, output_w;
	output_h = data_h + 2 * pad_h - kernel_h + 1;
	output_w = data_w + 2 * pad_w - kernel_w + 1;

	for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
	{
		for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
		{
			if (kernel[kernel_row * kernel_w + kernel_col] == 0)
			{
				continue;
			}

			int input_row = -pad_h + kernel_row;
			for (int output_rows = output_h; output_rows; output_rows--)
			{
				if (!(input_row >= 0 && input_row < data_h))
				{
					//逐列遍历输出数据，由于输入数据的行超出界限（补0)，对应的输出为0
					for (int output_cols = output_w; output_cols; output_cols--)
					{
						*(output++) = 0;
					}
				}
				else
				{
					int input_col = -pad_w + kernel_col;
					for (int output_col = output_w; output_col; output_col--)
					{
						//输入数据的行坐标和列坐标均没有超过界限
						if (input_col >= 0 && input_col < data_w)
						{
							*(output++) = data[input_row * data_w + input_col];
						}
						else
						{
							//如果输入列坐标超过界限，便置0
							*(output++) = 0;
						}
						input_col++;
					}
				}
				input_row++;
			}
		}
	}
}

template<typename DType>
void CalMat(DType *data_col, int data_h, int data_w,
	float *kernel, int kernel_count,
	DType *output)
{
	int rows = kernel_count;
	int cols = data_h*data_w;

	float temp = 0;
	for (int col = 0; col < cols; col++)
	{
		temp = 0;
		for (int row = 0; row < rows; row++)
		{
			temp += data_col[row*cols + col] * kernel[row];
		}
		*(output++) = temp;
	}
}

template<typename DType>
void DisplayMatrix(DType *data, int h, int w)
{
	for (int row = 0; row < h;row++)
	{
		for (int col = 0; col < w;col++)
		{
			printf("%d ",data[row * w + col]);
		}
		std::cout << std::endl;
	}
}

int * generateKernelMask(int scale, float orient, float delta, int & count)
{
	int kernel_h = 2 * scale + 1;
	int kernel_w = 2 * scale + 1;
	int *result = new int[kernel_h*kernel_w];
	memset(result, 0, sizeof(int)*kernel_h*kernel_w);
	count = 0;

	int xmin, xmax, ymin, ymax;
	float AL, AL1, AL2, COS_AL1, SIN_AL1, COS_AL2, SIN_AL2;
	AL = orient;
	AL1 = AL - delta;
	AL2 = AL + delta;

	COS_AL1 = cos(AL1);  SIN_AL1 = sin(AL1);
	COS_AL2 = cos(AL2);  SIN_AL2 = sin(AL2);
	xmin = (int)MIN(MIN(0, ceil(scale*COS_AL1)), ceil(scale*COS_AL2));
	xmax = (int)MAX(MAX(0, ceil(scale*COS_AL2)), ceil(scale*COS_AL1));
	ymin = (int)MIN(MIN(0, ceil(scale*SIN_AL2)), ceil(scale*SIN_AL1));
	ymax = (int)MAX(MAX(0, ceil(scale*SIN_AL1)), ceil(scale*SIN_AL2));
	
	for (int x = xmin; x <= xmax; x++)
	{
		for (int y = ymin; y <= ymax; y++)
		{
			// 点在sector内
			if ((y*y + x*x <= scale*scale)
				&& (y*COS_AL1 - x*SIN_AL1 - EPS >= 0)
				&& (y*COS_AL2 - x*SIN_AL2 + EPS <= 0))
			{
				result[(y + scale)*kernel_w + x + scale] = 1;
				count++;
			}
		}
	}

	return result;
}

template<typename DType>
DType * Convolution2D(DType *data, int data_h, int data_w,
	float *kernel, int kernel_h, int kernel_w)
{
	DType *output;
	int pad_h = kernel_h / 2;
	int pad_w = kernel_w / 2;

	int output_h, output_w;
	output_h = data_h + 2 * pad_h - kernel_h + 1;
	output_w = data_w + 2 * pad_w - kernel_w + 1;
	output = new DType[output_h*output_w];
	memset(output, 0, sizeof(DType) * output_h * output_w);

	for (int data_row = 0; data_row < data_h; data_row++)
	{
		for (int data_col = 0; data_col < data_w; data_col++)
		{
			float temp = 0;
			for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++)
			{
				for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++)
				{
					int kernel_index = kernel_row*kernel_w + kernel_col;
					int data_crow = data_row + kernel_row - pad_h;
					int data_ccol = data_col + kernel_col - pad_w;
					
					if ((data_crow >= 0 && data_crow < data_h) && (data_ccol >= 0 && data_ccol < data_w))
					{
						temp += kernel[kernel_index] * data[data_crow*data_w + data_ccol];
					}
				}
			}
			output[data_row*data_w + data_col] = temp;
		}
	}

	return output;
}

template<typename DType>
DType * OrientConv(DType *data, int data_h, int data_w,
	int scale, float orient, float delta)
{
	int kernel_h = 2 * scale + 1;
	int kernel_w = 2 * scale + 1;
	int kernel_count;
	int *kernel_mask = generateKernelMask(scale, orient, delta, kernel_count);

	float *kernel = new float[kernel_h*kernel_w];
	memset(kernel, 0, sizeof(float)*kernel_h*kernel_w);
	for (int i = 0; i < kernel_h*kernel_w; i++)
	{
		if (kernel_mask[i] != 0)
			kernel[i] = 1.0f / kernel_count;
	}
	DType *result = Convolution2D(data, data_h, data_w,
		kernel, kernel_h, kernel_w);

	//std::cout << "Convolution result:\n";
	//DisplayMatrix(result, output_h, output_w);

	delete[] kernel;
	delete[] kernel_mask;

	return result;
}

template<typename DType>
DType * Convolution2D_Fast(DType *data, int data_h, int data_w,
	float *kernel, int *kernel_mask, int kernel_h, int kernel_w, int kernel_count)
{
	DType *output;

	int pad_h = kernel_h / 2;
	int pad_w = kernel_w / 2;

	int output_h, output_w;
	output_h = data_h + 2 * pad_h - kernel_h + 1;
	output_w = data_w + 2 * pad_w - kernel_w + 1;
	DType *data_col = new DType[kernel_count*output_h*output_w];

	//std::cout << "data:\n";
	//DisplayMatrix(data, data_h, data_w);

	im2col(data, data_h, data_w, kernel_mask, kernel_h, kernel_w, data_col);
	//std::cout << "After im2col:\n";
	//DisplayMatrix(data_col, kernel_count, output_h*output_w);

	output = new DType[output_h*output_w];
	CalMat(data_col, data_h, data_w, kernel, kernel_count, output);

	delete[] data_col;

	return output;
}

// Calculate an orientational convolution for the input matrix data.
template<typename DType>
DType * OrientConv_Fast(DType *data, int data_h, int data_w,
	int scale, float orient, float delta)
{
	int kernel_h = 2 * scale + 1;
	int kernel_w = 2 * scale + 1;
	int kernel_count;
	int *kernel_mask = generateKernelMask(scale, orient, delta, kernel_count);
	float *kernel_fast = new float[kernel_count];
	for (int i = 0; i < kernel_count; i++)
	{
		kernel_fast[i] = 1.0f / kernel_count;
	}

	//std::cout << "kernel mask:\n";
	//DisplayMatrix(kernel_mask, kernel_h, kernel_w);

	DType *result_fast = Convolution2D_Fast(data, data_h, data_w,
		kernel_fast, kernel_mask, kernel_h, kernel_w, kernel_count);

	//std::cout << "Convolution result:\n";
	//DisplayMatrix(result_fast, data_h, data_w);

	delete[] kernel_fast;
	delete[] kernel_mask;

	return result_fast;
}

int _tmain(int argc, _TCHAR* argv[])
{
	//int data[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
	//int data_h = 4, data_w = 4;

	int scale = 7;
	int nbSector = (int)floor(2 * PI*scale);
	float deltaR = 2.5f;
	float widthSector = deltaR / scale;

	//int *result_fast = OrientConv_Fast(data, data_h, data_w, scale, orient, widthSector);

	//int kernel[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
	//int kernel_h = 3, kernel_w = 3;

	string str = "../images/test1.png";
	cv::Mat img = imread(str, 0);
	int data_h = img.rows, data_w = img.cols;
	

	float orient;
	double total = (double)getTickCount();
	for (int n = 0; n < nbSector;n++)
	{
		//double t = (double)getTickCount();
		orient = (float)2 * PI / (nbSector)*n;
		uchar *result_fast = OrientConv_Fast(img.data, data_h, data_w, scale, orient, widthSector);
		//t = ((double)getTickCount() - t) / getTickFrequency();
		//cout << "The " << n << "th convolution_fast costs " << t * 1000 << " milliseconds.\r\n";
	}
	total = ((double)getTickCount() - total) / getTickFrequency();
	cout << "Convolution_fast costs total " << total * 1000 << " milliseconds.\r\n";

	total = (double)getTickCount();
	for (int n = 0; n < nbSector; n++)
	{
		orient = (float)2 * PI / (nbSector)*n;
		uchar *result = OrientConv(img.data, data_h, data_w, scale, orient, widthSector);
	}
	total = ((double)getTickCount() - total) / getTickFrequency();
	cout << "Convolution costs total " << total * 1000 << " milliseconds.\r\n";
	
	//Mat img_result_fast;
	//img.copyTo(img_result_fast);
	//memcpy_s(img_result_fast.data, sizeof(uchar)*img.rows*img.cols, result_fast, sizeof(uchar)*img.rows*img.cols);

	//Mat img_result;
	//img.copyTo(img_result);
	//memcpy_s(img_result.data, sizeof(uchar)*img.rows*img.cols, result, sizeof(uchar)*img.rows*img.cols);

	//float error = 0;
	//for (int data_row = 0; data_row < data_h; data_row++)
	//{
	//	for (int data_col = 0; data_col < data_w; data_col++)
	//	{
	//		error += abs(result[data_row*data_w + data_col] - result_fast[data_row*data_w + data_col]);
	//	}
	//}

	//cout << "Error: " << error << endl;

	//imshow("origin", img);
	//imshow("result", img_result);
	//imshow("result_fast", img_result_fast);
	//waitKey(0);

	//delete[] result;
	//delete[] result_fast;

	return 0;
}

