#pragma once
#include <iostream>
#include <QDebug>
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
#include "QtLogger.h"
#include "config_parse.h"
#include "general_utils.h"


class DoSpecParamsAssess
{
public:
	DoSpecParamsAssess();

	DoSpecParamsAssess(ConfigParse* config);

	~DoSpecParamsAssess();

	int getSpecParamsRst(cv::Mat& frame, 
		std::map<std::string, std::vector<float>>& values, 
		std::map<std::string, cv::Mat>& resultPics, 
		std::map<std::string, int> classResult);

	int getSpecParamsRstV2(cv::Mat& frame,
		std::map<std::string, std::vector<float>>& values,
		std::map<std::string, cv::Mat>& resultPics,
		std::map<std::string, int> classResult);

private:
	MVEAAssess* m_pMveaAssesser;

	PVAssess* m_pPvAssesser;

	TDIIVSAssess* m_pTdiivsAssesser;

	TDIMVLWAssess* m_pTdimvlwAssesser;

	TRAssess* m_pTrAssesser;

	FVAVAssess* m_pFvavAssesser;
};

