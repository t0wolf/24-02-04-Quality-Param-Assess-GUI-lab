#include "DoFuncParamsAssess.h"


// 功能参数
DoFuncParamsAssess::DoFuncParamsAssess()
{
	std::string efSegEnginePath = "D:\\Resources\\20240221\\param_assess_models\\EF_seg_0306.engine";
	std::string efVideoEnginePath = "D:\\Resources\\20240221\\param_assess_models\\EF_video_0306.engine";
	//m_pEfAssesser = new EFAssess(efSegEnginePath, efVideoEnginePath);

	std::string lvvSegEnginePath = "D:\\Resources\\20240221\\param_assess_models\\LVV_seg_0416.engine";
	//m_pLvvAssesser = new LVVAssess(lvvSegEnginePath);

	std::string lvefEnginePath = "D:\\Resources\\20240221\\param_assess_models\\LVEF_biplane_0217.engine";

	m_biplaneLVEFAssesser = new BiplaneLVEFAssesser(lvefEnginePath);
	QtLogger::instance().logMessage("[I] Func LVEF Model Loaded");
}

DoFuncParamsAssess::DoFuncParamsAssess(ConfigParse* config)
{
	// 从配置文件中读取 LVEF 模型路径
	std::string lvefEnginePath;
	if (config->getSpecifiedNode("LVEF_BIPLANE_PATH", lvefEnginePath)) {

		// 检查 LVEF 模型路径是否存在
		if (GeneralUtils::fileExists(lvefEnginePath)) {
			m_biplaneLVEFAssesser = new BiplaneLVEFAssesser(lvefEnginePath);
			QtLogger::instance().logMessage(QString::fromStdString("[I] Func LVEF Model Loaded"));
		}
		else {
			QtLogger::instance().logMessage(QString::fromStdString("[E] LVEF model path does not exist: ") +
				QString::fromStdString(lvefEnginePath));
		}
	}
	else {
		QtLogger::instance().logMessage(QString::fromStdString("[E] Failed to load LVEF model path from config"));
	}
}

DoFuncParamsAssess::~DoFuncParamsAssess()
{
	delete m_pEfAssesser;
	delete m_pLvvAssesser;
	delete m_biplaneLVEFAssesser;
}

int DoFuncParamsAssess::getFuncParamsRst(std::vector<cv::Mat>& video, std::map<std::string,
	std::vector<float>>&values, std::map<std::string, cv::Mat>& resultPics)
{
	// LVV-edv  // pred1, pred2, curr(edv)
	if (video.size() < 3)
		return 0;

	std::vector<std::vector<cv::Mat>> edvFramesVec = { {video[1], video[0], video[2]}, {video[video.size() - 2], video[video.size() - 3], video.back()} };

	float edv = 0.0f;
	for (auto& edvFrames : edvFramesVec)
	{
		m_pLvvAssesser->lvvAssessment(edvFrames, values, resultPics);
		if (!values["LVV"].empty())
			edv += values["LVV"][0];
	}
	std::vector<float> edvVec = { edv / static_cast<float>(edvFramesVec.size()) };
	values.insert({ "EDV", edvVec });
	resultPics.insert({ "EDV", resultPics["LVV"] });
	values.erase("LVV");
	resultPics.erase("LVV");

	//LVV-esv  // 先减去video开头的edv的前两帧，找到中间位置的esv
	size_t esvIdx = video.size() / 2 + 1;
	std::vector<cv::Mat> esvFrames = { video[esvIdx - 1], video[esvIdx - 2], video[esvIdx] };
	m_pLvvAssesser->lvvAssessment(esvFrames, values, resultPics);
	values.insert({ "ESV", values["LVV"] });
	resultPics.insert({ "ESV", resultPics["LVV"] });
	values.erase("LVV");
	resultPics.erase("LVV");

	// EF
	//m_pEfAssesser->efAssessment(video, values, resultPics);
	//m_biplaneLVEFAssesser->doInference()

	return 0;
}

int DoFuncParamsAssess::getFuncParamsRst(std::string& viewName, std::vector<cv::Mat>& video, std::vector<int> vKeyframeIdxes, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	//if (video.size() < 10)
	//{
	//	QtLogger::instance().logMessage("[E] Func Params - Video size less than 10.");
	//	return 0;
	//}

	int ret = m_biplaneLVEFAssesser->doInference(viewName, video, vKeyframeIdxes, values, resultPics);

	return ret;
}
