#pragma once

#include "segment_infer_base.h"

class LVVSegmentInferer : public SegmentInferBase
{
public:
	LVVSegmentInferer(std::string& sEngineFilePath);

	~LVVSegmentInferer();

	int doInference(std::vector<cv::Mat>& video, std::vector<cv::Mat>& vMasks);

	int preprocess(std::vector<cv::Mat>& video, float* blob);

	int blobFromImage(std::vector<cv::Mat>& video, float* blob);

private:
	int doSingleInfer(std::vector<cv::Mat>& video, std::vector<cv::Mat>& vMasks);

	int postProcess(float* output, std::vector<cv::Mat>& vMasks);

};