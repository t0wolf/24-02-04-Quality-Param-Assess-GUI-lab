#pragma once
#include <iostream>
#include "ef_assess.h"
#include "lvv_assess.h"
#include "BiplaneLVEFAssesser.h"
#include "QtLogger.h"
#include "config_parse.h"


class DoFuncParamsAssess
{
public:
	DoFuncParamsAssess();

	DoFuncParamsAssess(ConfigParse* config);

	~DoFuncParamsAssess();

	int getFuncParamsRst(std::vector<cv::Mat>& video, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	int getFuncParamsRst(std::string& viewName, std::vector<cv::Mat>& video, std::vector<int> vKeyframeIdxes,
		std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

	inline void clearLVEFDataCache()
	{
		m_biplaneLVEFAssesser->clearBiplaneVideo();
	}

	inline bool isFuncBiplaneMode()
	{
		return m_biplaneLVEFAssesser->isBiplaneMode();
	}

	inline void setScaleInfo(float scaleLength, float scale)
	{
		m_biplaneLVEFAssesser->setScaleInfo(scaleLength, scale);
	}

private:
	EFAssess* m_pEfAssesser;

	LVVAssess* m_pLvvAssesser;

	BiplaneLVEFAssesser* m_biplaneLVEFAssesser;
};
