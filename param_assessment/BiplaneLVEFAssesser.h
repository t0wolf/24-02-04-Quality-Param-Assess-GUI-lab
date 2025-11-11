#pragma once
//#include <filesystem>
#include <QFile>
#include "LVEFAssesser.h"
#include "simpson_calculate.h"
#include "QtLogger.h"
#include "general_utils.h"

//namespace fs = std::filesystem;

class BiplaneLVEFAssesser
{
public:
	BiplaneLVEFAssesser(std::string& lvefSegEnginePath);
	~BiplaneLVEFAssesser();

	int doInference(std::string& currViewName, std::vector<cv::Mat>& vVideoClips, std::vector<int>& keyFrameIndex,
		std::map<std::string, std::vector<float>>&values, std::map<std::string, cv::Mat>& resultPics);

	inline void clearBiplaneVideo()
	{
		m_a2cVideo.clear();
		m_a4cVideo.clear();

		m_a2cHistVolumeInfo = std::pair<VolumeInfo, VolumeInfo>{};
		m_a4cHistVolumeInfo = std::pair<VolumeInfo, VolumeInfo>{};

		m_bIsBiplaneMode = false;
		m_fPixPerUnit = -10000.0f;

		clearAllHistoryRecords();
	}

	inline bool isBiplaneMode()
	{
		return m_bIsBiplaneMode;
	}

	inline void setScaleInfo(float scaleLength, float scale)
	{
		QMutexLocker locker(&m_scaleMutex);
		m_fPixPerUnit = scale;
		m_fLength = scaleLength;
	}

private:
	int biplaneSimpson(cv::Mat& a2cEdMask, 
		cv::Mat& a2cEsMask,
		cv::Mat& a4cEdMask,
		cv::Mat& a4cEsMask);

	int parseKeyframes(std::vector<int>& keyframeIdx, std::vector<cv::Mat>& inputVideo, std::vector<cv::Mat>& outputVideo, int length);

	cv::Mat concatMultiImages(std::vector<cv::Mat>& vecImages, cv::Size& targetSize);

	bool checkFinalResultsInRange(std::map<std::string, std::vector<float>>& values);

	bool checkLVEFResultsInRange(float fValue);

	std::vector<std::string> strSplitSpace(std::string& strInput);

	int clearHistoryBuffers(std::string& strCurrViewName);

	//int updateHistoryVolumeInfo(std::string& strCurrViewName, VolumeInfo& currVolumeInfo);

	int updateHistoryVolumeInfo(std::string& strCurrViewName, std::pair<VolumeInfo, VolumeInfo>& currVolumeInfo);

	int updateHistoryPremiums(std::string& strCurrViewName, std::vector<cv::Mat>& vecPremiums);

	std::vector<cv::Mat> getHistoryPremiums(std::string& strCurrViewName);

	int clearHistoryPremiums(std::string& strCurrViewName);

	int clearAllHistoryRecords();
	
private:
	LVEFAssesser* m_lvefAssesser;
	std::string m_lvefSegEnginePath;

	int m_viewFrameLength;
	bool m_bIsBiplaneMode;

	std::vector<cv::Mat> m_a2cVideo, m_a4cVideo;
	//VolumeInfo m_a2cHistVolumeInfo, m_a4cHistVolumeInfo;
	std::pair<VolumeInfo, VolumeInfo> m_a2cHistVolumeInfo, m_a4cHistVolumeInfo;
	std::vector<cv::Mat> m_a2cHistPremiums, m_a4cHistPremiums;

	QMutex m_scaleMutex;

	std::map<std::string, std::pair<float, float>> m_referRange = {
		{ "EF", {25.0f, 80.0f} }
	};

	float m_fPixPerUnit, m_fLength;

	QString m_videoSaveRootPath = "D:/Data/Saved-Images/B-Mode";
};

