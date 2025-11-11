#pragma once

#include "segment_infer_base.h"

class AAOSegmentInferer : public SegmentInferBase
{
public:
	AAOSegmentInferer(std::string& sEngineFilePath);

	~AAOSegmentInferer();

	int doInference(cv::Mat& src, std::vector<cv::Mat>& vMasks);

private:
	int doSingleInfer(cv::Mat& src, std::vector<cv::Mat>& vMasks);

	int postProcess(float* output, std::vector<cv::Mat>& vMasks);
};

