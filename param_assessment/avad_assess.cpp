#include "avad_assess.h"

AVADAssess::AVADAssess(std::string& sEngineFilePath)
	: m_keyptEnginePath(sEngineFilePath)
	, m_scale(-10000.0f)
{
	m_avadKeyptInferer = new AVADKeypointsInferer(m_keyptEnginePath);
}

AVADAssess::~AVADAssess()
{
	delete m_avadKeyptInferer;
}

int AVADAssess::doKeyptInference(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	m_avadKeyptInferer->doInference(src, values, resultPics, m_scale);
	return 1;
}

int AVADAssess::avadAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	//std::vector<cv::Mat> vMasks;
	doKeyptInference(src, values, resultPics);

	return 1;
}
int AVADAssess::setScaleInfo(float& scaleLength, float& scale)
{
	m_scaleLength = scaleLength;
	m_scale = scale;
	return 1;
}