#pragma once
#include <numeric>
#include "ocr_inference/OCRInferer.h"

class ScaleExtractor
{
public:
	ScaleExtractor();
	~ScaleExtractor();

	float extractScale(cv::Mat& src);

private:
	int extractText(cv::Mat& src, std::vector<cv::Rect>& vTextROIs, std::vector<std::string>& vTexts);

	float getRealScale(std::vector<std::string>& vTexts, std::vector<cv::Rect>& vTextROIs);

private:
	OCRInferer* m_ocrInferer;
};

