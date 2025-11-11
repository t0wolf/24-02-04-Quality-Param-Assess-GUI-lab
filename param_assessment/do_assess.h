#pragma once
#include <iostream>
//#include <filesystem>
#include "segment_infer_base.h"
#include "ivs_and_pw_assess.h"
#include "aao_segment_inferer.h"
#include "aao_assess.h"
#include "aad_assess.h"
#include "asd_and_sjd_access.h"
#include "avad_assess.h"
#include "avad_keypoints_inferer.h"
#include "laad_assess.h"
#include "ef_assess.h"
#include "mvea_assess.h"
#include "pv_assess.h"
#include "tdi_ivs_assess.h"
#include "tdi_mvlw_assess.h"
#include "tr_assess.h"
#include "fvav_assess.h"
#include "spec_classification_inferer.h"
#include "lvv_assess.h"

class DoStrucParamsAssess 
{
public:
	DoStrucParamsAssess();

	~DoStrucParamsAssess();

	int getStrucParamsRst(std::vector<cv::Mat>& video, std::vector<int>& keyframeIdx, std::map<std::string,
		std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPic);

private:
	cv::Mat m_inferFrame;

	IVSAndPWAssess* m_pIvsandpwAssesser;

	AVADAssess* m_pAvadAssesser;

	ASDAndSJDAssess* m_pAsdAndsjdAssesser;

	AADAssess* m_pAadAssesser;

	LAADAssess* m_pLaadAssesser;
};

class DoFuncParamsAssess 
{
public:
	DoFuncParamsAssess();

	~DoFuncParamsAssess();

	int getFuncParamsRst(std::vector<cv::Mat>& video, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

private:
	EFAssess* m_pEfAssesser;

	LVVAssess* m_pLvvAssesser;
};

class DoSpecParamsAssess
{
public:
	DoSpecParamsAssess();

	~DoSpecParamsAssess();

	int getSpecParamsRst(cv::Mat& frame, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics);

private:
	SPECClassInferer* m_pSpecInferer;

	MVEAAssess* m_pMveaAssesser;

	PVAssess* m_pPvAssesser;

	TDIIVSAssess* m_pTdiivsAssesser;

	TDIMVLWAssess* m_pTdimvlwAssesser;

	TRAssess* m_pTrAssesser;

	FVAVAssess* m_pFvavAssesser;
};