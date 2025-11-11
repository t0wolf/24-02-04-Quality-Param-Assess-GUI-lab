#include "fvav_assess.h"

FVAVAssess::FVAVAssess(std::string& sEngineFilePath)
	: m_segEnginePath(sEngineFilePath)
{
	m_aaoSegInferer = new AAOSegmentInferer(m_segEnginePath);
}

FVAVAssess::~FVAVAssess()
{
	delete m_aaoSegInferer;
}

int FVAVAssess::doSegInference(cv::Mat& src, std::vector<cv::Mat>& vMasks)
{
	m_aaoSegInferer->doInference(src, vMasks);
	return 1;
}

int FVAVAssess::fvavAssessment(cv::Mat& src, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<cv::Mat> vMasks;
	doSegInference(src, vMasks);
	postProcess(src, vMasks, values, resultPics);

	return 1;
}

int FVAVAssess::postProcess(cv::Mat& src, std::vector<cv::Mat>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<std::vector<cv::Point>> maskContours;
	cv::Mat fvavMask = vMasks[0];
	cv::Size originSize(src.cols, src.rows);
	cv::resize(fvavMask, fvavMask, originSize);  // 将mask映射回原图的大小，便于后续比例尺直接应用
	//cv::findContours(fvavMask, maskContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	//ImageProcess::filterSmallContour(maskContours);
	//std::vector<cv::Point> maskContour = maskContours[0];

	// # 漫水填充从(0, 0)点开始,再反转和原图求交集，两次闭运算，得到波形的轮廓
	cv::Mat maskFlood = fvavMask.clone();
	cv::Mat maskCopy = fvavMask.clone();
	cv::Mat	maskFill = cv::Mat::zeros(fvavMask.rows + 2, fvavMask.cols + 2, CV_8UC1);
	cv::Mat maskFloodInv;

	cv::floodFill(maskFlood, maskFill, cv::Point(0, 0), cv::Scalar(255));
	cv::bitwise_not(maskFlood, maskFloodInv);
	cv::Mat imgOut = maskCopy | maskFloodInv;

	cv::Mat closeOneImg, closeTwoImg;
	cv::Mat kernalOne = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));
	cv::Mat kernalTwo = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
	cv::morphologyEx(imgOut, closeOneImg, cv::MORPH_CLOSE, kernalOne);
	cv::morphologyEx(closeOneImg, closeTwoImg, cv::MORPH_CLOSE, kernalOne);

	cv::Mat colorMask;
	cv::applyColorMap(closeTwoImg, colorMask, cv::COLORMAP_JET);
	//cv::Mat dst;
	//cv::addWeighted(src, 1.0, colorMask, 0.5, 0.0, dst);
	//cv::imshow("Ao", dst);
	//cv::waitKey(0);

	cv::findContours(closeTwoImg, maskContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	if (maskContours.size() < 2)
		return 0;

	// 计算每个波形的面积，最低点和最高点
	std::vector<float> maskArea;
	std::vector<int> topY, bottomY, vAoValues;  // 下点和上点
	cv::Mat drawImg = src.clone();

	for (auto& maskContour : maskContours)
	{
		maskArea.push_back(cv::contourArea(maskContour));
		std::vector<cv::Point> sortedContour(maskContour);
		std::sort(sortedContour.begin(), sortedContour.end(), [](cv::Point a, cv::Point b) {
			return a.y < b.y;
			});
		topY.push_back(sortedContour.back().y);
		bottomY.push_back(sortedContour[0].y);
		vAoValues.push_back(std::abs(sortedContour[0].y - sortedContour.back().y));
		//cv::drawContours(drawImg, maskContour, -1, cv::Scalar(0, 255, 0), 2);
		cv::circle(drawImg, sortedContour.back(), 5, cv::Scalar(0, 0, 255), -1);
	}
	//cv::drawContours(drawImg, maskContours, -1, cv::Scalar(0, 255, 0), 2);

	float scale = 1.0f;
	std::vector<float> meanArea;
	meanArea.push_back(std::accumulate(maskArea.begin(), maskArea.end(), 0) / static_cast<float>(maskContours.size()));
	int meanBottom = std::accumulate(bottomY.begin(), bottomY.end(), 0) / maskContours.size();
	int meanTop = std::accumulate(topY.begin(), topY.end(), 0) / maskContours.size();

	//int pointsSize = topY.size();
	//int nOutlier = 0;
	//topY = ParamsAssessUtils::filterOutliers(topY, 1.5, nOutlier);

	//int currThresh = static_cast<int>(std::floor(static_cast<float>(pointsSize) * 0.4f));
	//if (nOutlier >= currThresh)
	//	return 0;

	//int medianTop = ParamsAssessUtils::findMedium(topY);
	////int maxTop = *(std::max_element(topY.begin(), topY.end()));
	//std::vector<float> avVelocity = { (medianTop - meanBottom) * scale };
	////std::vector<float> avVelocity = { (maxTop) * scale };

	//values.insert({ "Ao", avVelocity });
	//values.insert({ "VIT", meanArea });
	//resultPics.insert({ "Ao", drawImg });

	ParamsAssessUtils::generateFinalResult("Ao", vAoValues, values, drawImg, resultPics);

	return 1;
}


