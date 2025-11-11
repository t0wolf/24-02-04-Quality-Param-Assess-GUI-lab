//
// Created by Ziyon Zeng && Hanlin Cheng on 2022/7/20.
//
//#include "opencv.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <math.h>
#include <algorithm>
#include <cstdlib>
#include <numeric>

class RoIExtract {
public:
	RoIExtract();

	~RoIExtract();

	void preprocessVideo(std::vector<cv::Mat> frames_gray, const int totalCount, cv::Mat& realmask, std::vector<float>& radius, cv::Rect& croprect);

private:
    void setInput(std::vector<cv::Mat> frames_gray, const int totalCount);
	void countMask(float gap, cv::Mat &realmask, cv::Point &crosspt, std::vector<float> &radius, bool firsttime);
	void houghPostProcessing(int faultflag, cv::Point &crosspt, std::vector<cv::Vec2f> lines, std::vector<float> &radius);
	std::vector<float> findPeakvalue(std::vector<float> num, int count);
	void drawMask(cv::Mat &realmask, cv::Point crosspt, std::vector<float> &radius, std::vector<cv::Vec4i> lines);
	void getCropPoints(cv::Mat realmask, cv::Vec4i &points);

	std::vector<cv::Mat> m_framesGray;
	cv::Mat m_maskTemp;
	cv::Mat m_maskAll;
};