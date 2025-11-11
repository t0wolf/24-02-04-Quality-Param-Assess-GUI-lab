#pragma once

#include "segment_infer_base.h"

class EFSegmentInferer : public SegmentInferBase
{
public:
	EFSegmentInferer(std::string& sEngineFilePath);

	~EFSegmentInferer();

	int doInference(std::vector<cv::Mat>& video, std::vector<int>& framePixels, std::vector<cv::Mat>& frameMasks);

	int preprocess(cv::Mat& src, float* blob);

	int blobFromImage(cv::Mat& src, float* blob);

private:
	int doSingleInfer(cv::Mat& src, int& laPixels, cv::Mat& lvMasks);

	int postProcess(float* output, int& laPixels, cv::Mat& lvMasks);
};