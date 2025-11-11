#include "fvav_assess.h"

FVAVAssess::FVAVAssess(std::string& sEngineFilePath)
	: m_segEnginePath(sEngineFilePath)
{
	m_aoVTISegInferer = new AoVTISegmentInferer(m_segEnginePath);
}

FVAVAssess::~FVAVAssess()
{
    if (m_aoVTISegInferer != nullptr)
	    delete m_aoVTISegInferer;
}

int FVAVAssess::doSegInference(cv::Mat& src, std::vector<cv::Mat>& vMasks)
{
	m_aoVTISegInferer->doInference(src, vMasks);
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
    // 创建一个用于存储轮廓的vector
    std::vector<std::vector<cv::Point>> maskContours;

    // 取出第一个mask，并将其调整为原图大小
    cv::Mat fvavMask = vMasks[0];
    cv::Size originSize(src.cols, src.rows);
    cv::resize(fvavMask, fvavMask, originSize);  // 将mask映射回原图的大小，便于后续比例尺直接应用

    //cv::imshow("test", fvavMask);
    //cv::waitKey(0);

    // 克隆mask，用于后续处理
    cv::Mat maskFlood = fvavMask.clone();
    cv::Mat maskCopy = fvavMask.clone();
    cv::Mat maskFill = cv::Mat::zeros(fvavMask.rows + 2, fvavMask.cols + 2, CV_8UC1);
    cv::Mat maskFloodInv;

    // 漫水填充从(0, 0)点开始
    cv::floodFill(maskFlood, maskFill, cv::Point(0, 0), cv::Scalar(255));

    // 取反得到maskFlood的反转图像
    cv::bitwise_not(maskFlood, maskFloodInv);

    // 原mask和反转mask的并集
    cv::Mat imgOut = maskCopy | maskFloodInv;

    // 进行两次闭运算
    cv::Mat closeOneImg, closeTwoImg;
    cv::Mat kernalOne = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(8, 8));
    cv::Mat kernalTwo = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
    cv::morphologyEx(imgOut, closeOneImg, cv::MORPH_CLOSE, kernalOne);
    cv::morphologyEx(closeOneImg, closeTwoImg, cv::MORPH_CLOSE, kernalOne);

    // 查找轮廓
    cv::findContours(closeTwoImg, maskContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (maskContours.empty())
        return 0;

    // 计算每个波形的面积，最低点和最高点
    std::vector<float> maskArea, aoVelocities;
    std::vector<int> topY, bottomY;  // 下点和上点
    
    cv::Mat drawImg = src.clone();

    for (auto& maskContour : maskContours)
    {
        maskArea.push_back(cv::contourArea(maskContour));

        // 按Y坐标排序，找到最高和最低点
        std::vector<cv::Point> sortedContour(maskContour);
        std::sort(sortedContour.begin(), sortedContour.end(), [](cv::Point a, cv::Point b) {
            return a.y < b.y;
            });
        topY.push_back(sortedContour.back().y);
        bottomY.push_back(sortedContour[0].y);

        // 在最高点处画一个圆
        cv::circle(drawImg, sortedContour.back(), 5, cv::Scalar(0, 0, 255), -1);
    }

    // 计算平均面积、平均高度差
    float scale = 1.0f;
    std::vector<float> meanArea;
    meanArea.push_back(std::accumulate(maskArea.begin(), maskArea.end(), 0) / static_cast<float>(maskContours.size()));
    int meanBottom = std::accumulate(bottomY.begin(), bottomY.end(), 0) / maskContours.size();

    for (auto& top : topY)
    {
        float currV = static_cast<float>(top - meanBottom);
        aoVelocities.push_back(currV);
    }

    int meanTop = std::accumulate(topY.begin(), topY.end(), 0) / maskContours.size();
    std::vector<float> avVelocity = { (meanTop - meanBottom) * scale };

    // 将计算结果存入values中
    values.insert({ "Ao", aoVelocities });
    values.insert({ "VTI", meanArea });
    // 将处理后的图像存入resultPics中
    resultPics.insert({ "Ao", drawImg });

    return 0;
}
