#pragma once

#include "image_process.h"
#include "assess_utils.h"
#include "segment_infer_base.h"

class LVEFAssesser : public SegmentInferBase
{
public:
	LVEFAssesser(std::string& sEnginePath);
	~LVEFAssesser();

	int doInference(std::vector<cv::Mat>& a2cVideo, std::vector<cv::Mat>& a4cVideo, std::vector<std::vector<cv::Mat>>& vMasks);

private:
	int preprocess(std::vector<cv::Mat>& video, float* blob);

	int blobFromImage(std::vector<cv::Mat>& video, float* blob);

	int doSingleInfer(std::vector<cv::Mat>& a2cVideo, std::vector<cv::Mat>& a4cVideo, std::vector<std::vector<cv::Mat>>& vMasks);

	int postProcess(float* outputMask, std::vector<cv::Mat>& vMasks);

private:
	int m_aggVideoLength;

	int m_inputA2CIdx, m_inputA4CIdx, m_outputA4CEDIdx, m_outputA4CESIdx, m_outputA2CEDIdx, m_outputA2CESIdx;

	nvinfer1::Dims4 m_outputMaskDims;
};
