#pragma once

#include "segment_infer_base.h"

class EFVideoInferer : public SegmentInferBase
{
public:
	EFVideoInferer(std::string& sEngineFilePath);

	~EFVideoInferer();

	int doInference(std::vector<cv::Mat>& video, std::vector<float>& predScores);

	int clipVideo(std::vector<cv::Mat>& srcVideo, std::vector<std::vector<cv::Mat>>& clipedVideos);

	int preprocess(std::vector<cv::Mat>& video, float* blob);

	int blobFromImage(std::vector<cv::Mat>& video, float* blob);

private:
	int doSingleInfer(std::vector<cv::Mat>& video, float& score);

	int postProcess(float* output, float& score);

	nvinfer1::Dims32 m_inputDims;
	nvinfer1::Dims32 m_outputDims;
};