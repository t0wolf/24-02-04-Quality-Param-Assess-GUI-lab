/*
   Created by SEU-BME-LBMD-chl, SEU-BME-LBMD-zzy, SEU-BME-LBMD-scj
*/
#pragma once
//#include "opencv.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"

const short Piexl = 255;                 //像素值
const short Piexl_T = 25;                //像素阈值


std::vector<cv::Mat> getFramesRect(std::vector<cv::Mat> img_r, cv::Rect rect);
std::vector<cv::Mat> getFramesResize(std::vector<cv::Mat> img_r, int new_size);
std::vector<cv::Mat> getFramesResizeWH(std::vector<cv::Mat> img_r, int new_size);
std::vector<cv::Mat> getFramesBGR2RGB(std::vector<cv::Mat> img_r);
std::vector<cv::Mat> getFramesMaskedData(std::vector<cv::Mat> img_r, cv::Mat img_mask);
std::vector<int> genSampleIdxs(std::vector<cv::Mat> img_r, int frameSampleRate);
std::vector<cv::Mat> getFramesSampled(std::vector<cv::Mat> img_r, std::vector<int> sampleIdxs);

//计算图像直方图
class HistogramMat
{
private:
	int histSize[1];
	float hranges[2];
	const float* ranges[1];
	int channels[1];

public:
	HistogramMat();
	cv::Mat getHistogram(const cv::Mat& image, const cv::Mat mask);
};