#include "ed_es_line_assess.h"

MultiLineAssess::MultiLineAssess(std::string& sEnginePath, std::string& sDetachTwoEnginePath, std::string& sDetachFourEnginePath, std::string& sMultiLineMaskPath)
	:m_heatmapPath(sEnginePath)
    ,m_detachTwoHeatmapPath(sDetachTwoEnginePath)
    ,m_detachFourHeatmapPath(sDetachFourEnginePath)
    ,m_multiLineMaskPath(sMultiLineMaskPath)
    ,m_scale(-10000.0f)
    ,m_scaleLength(0.0f)
{
	m_heatmapInferer = new LineInferer(m_heatmapPath);
    m_detachTwoHeatmapInferer = new DetachTwoLineInferer(m_detachTwoHeatmapPath);
    m_detachFourHeatmapInferer = new DetachFourLineInferer(m_detachFourHeatmapPath);
    m_multiLineMaskInferer = new MultiLineMaskInferer(m_multiLineMaskPath);
}

MultiLineAssess::~MultiLineAssess()
{
	delete m_heatmapInferer;
    delete m_detachTwoHeatmapInferer;
    delete m_detachFourHeatmapInferer;
    delete m_multiLineMaskInferer;
}

int MultiLineAssess::doHeatmapInference(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks)
{
	m_heatmapInferer->doInference(video, vMasks);
	return 1;
}

int MultiLineAssess::doTwoHeatmapInference(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks)
{
    m_detachTwoHeatmapInferer->doInference(video, vMasks);
    return 1;
}

int MultiLineAssess::doFourHeatmapInference(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks)
{
    m_detachFourHeatmapInferer->doInference(video, vMasks);
    return 1;
}

int MultiLineAssess::doMultiLineMaskInference(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks)
{
    m_multiLineMaskInferer->doInference(video, vMasks);
    return 1;
}

int MultiLineAssess::multiLineAssessment(std::vector<cv::Mat>& video, std::map<std::string, std::vector<float>>& values,
    std::map<std::string, cv::Mat>& resultPics)
{
	std::vector<std::vector<cv::Mat>> vMasks;
	doHeatmapInference(video, vMasks);

	postProcess(video, vMasks, values, resultPics);
	return 1;
}

int MultiLineAssess::detachLineAssessment(std::vector<cv::Mat>& video, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
    std::vector<std::vector<cv::Mat>> vTwoMasks, vFourMasks;
    doTwoHeatmapInference(video, vTwoMasks);
    postProcessTwoHeatmap(video, vTwoMasks, values, resultPics);

    doFourHeatmapInference(video, vFourMasks);
    postProcessFourHeatmap(video, vFourMasks, values, resultPics);

    return 1;
}

int MultiLineAssess::multiLineMaskAssessment(std::vector<cv::Mat>& video, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
    std::vector<std::vector<cv::Mat>> vMasks;
    doMultiLineMaskInference(video, vMasks);
    postProcessMultiLineMask(video, vMasks, values, resultPics);
    return 1;
}

int MultiLineAssess::postProcess(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{

    std::vector<std::vector<cv::Point2f>> vKeypoints;
    std::vector<float> vDistances;

    cv::Size originalImageSize = video[0].size();
    cv::Size currImageSize = vMasks[0][0].size();

    float scaleH = static_cast<float>(originalImageSize.height) / static_cast<float>(currImageSize.height);
    float scaleW = static_cast<float>(originalImageSize.width) / static_cast<float>(currImageSize.width);


    for (size_t i = 0; i < vMasks.size(); ++i) 
    {
        for (size_t j = 0; j < vMasks[i].size(); ++j) 
        {
            cv::Mat& heatmap = vMasks[i][j];
            cv::Mat normalizedMap, binaryMap;
            std::vector<cv::Point2f> keypoints;

            //cv::threshold(mat, mat, 0, 255, cv::THRESH_TOZERO);

            cv::normalize(heatmap, normalizedMap, 0, 255, cv::NORM_MINMAX);  // 归一化到0-255
            normalizedMap.convertTo(normalizedMap, CV_8UC1);

            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            int thresh;

            // 查找Mat中的最小值和最大值及其位置
            cv::minMaxLoc(normalizedMap, &minVal, &maxVal, &minLoc, &maxLoc);

            if (maxVal <= 10)
                thresh = static_cast<int>(maxVal);
            else
                thresh = static_cast<int>(0.5 * maxVal);
            cv::threshold(normalizedMap, binaryMap, thresh, 255, cv::THRESH_BINARY);
            
            //if (i == 0 && (j == 3 || j == 4))  // ivs和pw单独处理
            //{
            //    cv::threshold(normalizedMap, binaryMap, 100, 255, cv::THRESH_BINARY);  // 二值化
            //}
            //else 
            //{
            //    cv::threshold(normalizedMap, binaryMap, 50, 255, cv::THRESH_BINARY);
            //}

            // for test
            //cv::imshow("test", binaryMap);
            //cv::waitKey(0);
            // for test

            std::vector<std::vector<cv::Point>> maskContours;
            cv::findContours(binaryMap, maskContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (maskContours.size() == 0) 
            {
                vKeypoints.push_back({ cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f) });
                vDistances.push_back(0.f);
                continue;
            }

            std::vector<cv::Point> maskContour = ImageProcess::findMaxContour(maskContours);
            if (maskContour.empty()) 
            {
                vKeypoints.push_back({ cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f) });
                vDistances.push_back(0.f);
                continue;
            }
            cv::RotatedRect maskContourRect = cv::minAreaRect(maskContour);

            cv::Point2f rectPoints[4];
            maskContourRect.points(rectPoints);

            // for test
            //cv::Mat outputImage = cv::Mat::zeros(binaryMap.size(), CV_8UC3);
            //cv::cvtColor(binaryMap, outputImage, cv::COLOR_GRAY2BGR);
            //for (int j = 0; j < 4; j++) 
            //{
            //    cv::line(outputImage, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
            //}
            //cv::imshow("test", outputImage);
            //cv::waitKey(0);
            // for test

            std::sort(rectPoints, rectPoints + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
                return a.y < b.y;});

            if (i == 0 && (j == 3 || j == 4))  // ivs和pw对点重新进行调整
            {
                processPoints(rectPoints);
                // for test
                //cv::Mat dstImageEd = video[0].clone();

                //cv::Point testPoints[4];
                //for (int i = 0; i < 4; i++) 
                //{
                //    testPoints[i].x = static_cast<int>(rectPoints[i].x * scaleW);
                //    testPoints[i].y = static_cast<int>(rectPoints[i].y * scaleH);
                //}
                //for (auto& point : testPoints) 
                //{
                //    cv::circle(dstImageEd, point, 2, (0, 0, 255), -1);
                //}
                //cv::imshow("test", dstImageEd);
                //cv::waitKey(0);
                // for test
            }

            cv::Point2f upPoint, downPoint;  // 映射回原图
            upPoint.x = (rectPoints[0].x + rectPoints[1].x) * 0.5 * scaleW;
            upPoint.y = (rectPoints[0].y + rectPoints[1].y) * 0.5 * scaleH;
            downPoint.x = (rectPoints[2].x + rectPoints[3].x) * 0.5 * scaleW;
            downPoint.y = (rectPoints[2].y + rectPoints[3].y) * 0.5 * scaleH;
            
            keypoints.assign({ upPoint, downPoint });
            vKeypoints.push_back(keypoints);
            keypoints.clear();

            float distance = ParamsAssessUtils::calcLineDist(upPoint, downPoint);
            vDistances.push_back(distance);
        }
    }

    //std::vector<std::string> vClassNames = { "Sinus of Valsalvad", "Sinotubular Junction", "Ascending Aorta", "IVSd", "LVPWd", "LVIDd", "LAAd" };
    std::vector<std::string> vClassNames = { "ASD", "SJD", "AAD", "IVSTd", "LVPWTd", "LVDd", "LAD" };

    for (int i = 0; i < vClassNames.size(); i++) 
    {
        if (i <= 5) 
        {
            cv::Mat dstImage = video[0].clone();
            std::vector<float> dist = { vDistances[i] };
            cv::line(dstImage, vKeypoints[i][0], vKeypoints[i][1], cv::Scalar(0, 255, 0), 2);
            //cv::imshow("test", dstImage);
            //cv::waitKey(0);
            resultPics.emplace(vClassNames[i], dstImage);
            values.emplace(vClassNames[i], dist);
        }
        else 
        {
            cv::Mat dstImage = video.back().clone();
            std::vector<float> dist = { vDistances[i] };
            cv::line(dstImage, vKeypoints[i][0], vKeypoints[i][1], cv::Scalar(0, 255, 0), 2);
            //cv::imshow("test", dstImage);
            //cv::waitKey(0);
            resultPics.emplace(vClassNames[i], dstImage);
            values.emplace(vClassNames[i], dist);
        }
    }

	return 0;
}

int MultiLineAssess::postProcessTwoHeatmap(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
    std::vector<std::vector<cv::Point2f>> vKeypoints, vRectPoints;
    std::vector<float> vDistances;
    std::vector<cv::Mat> vRedMasks;

    cv::Size originalImageSize = video[0].size();
    cv::Size currImageSize = vMasks[0][0].size();

    float scaleH = static_cast<float>(originalImageSize.height) / static_cast<float>(currImageSize.height);
    float scaleW = static_cast<float>(originalImageSize.width) / static_cast<float>(currImageSize.width);


    for (size_t i = 0; i < vMasks.size(); ++i)
    {
        for (size_t j = 0; j < vMasks[i].size(); ++j)
        {
            cv::Mat& heatmap = vMasks[i][j];
            cv::Mat normalizedMap, binaryMap;
            std::vector<cv::Point2f> keypoints;

            //cv::threshold(mat, mat, 0, 255, cv::THRESH_TOZERO);

            cv::normalize(heatmap, normalizedMap, 0, 255, cv::NORM_MINMAX);  // 归一化到0-255
            normalizedMap.convertTo(normalizedMap, CV_8UC1);

            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            int thresh;

            // 查找Mat中的最小值和最大值及其位置
            cv::minMaxLoc(normalizedMap, &minVal, &maxVal, &minLoc, &maxLoc);

            if (maxVal <= 10)
                thresh = static_cast<int>(maxVal);
            else
                thresh = static_cast<int>(0.5 * maxVal);
            cv::threshold(normalizedMap, binaryMap, thresh, 255, cv::THRESH_BINARY);

            // for test
            //cv::imshow("test", binaryMap);
            //cv::waitKey(0);
            // for test

            cv::Mat redMask;
            std::vector<cv::Mat> channels = { cv::Mat::zeros(binaryMap.size(), CV_8UC1),  // Blue channel (0)
                                             cv::Mat::zeros(binaryMap.size(), CV_8UC1),  // Green channel (0)
                                             binaryMap };                                 // Red channel

            cv::merge(channels, redMask);

            std::vector<std::vector<cv::Point>> maskContours;
            cv::findContours(binaryMap, maskContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (maskContours.size() == 0)
            {
                vKeypoints.push_back({ cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f) });
                vDistances.push_back(0.f);
                vRedMasks.push_back(redMask);
                vRectPoints.push_back({ cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f) });
                continue;
            }

            std::vector<cv::Point> maskContour = ImageProcess::findMaxContour(maskContours);
            if (maskContour.empty())
            {
                vKeypoints.push_back({ cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f) });
                vDistances.push_back(0.f);
                vRedMasks.push_back(redMask);
                vRectPoints.push_back({ cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f) });
                continue;
            }
            cv::RotatedRect maskContourRect = cv::minAreaRect(maskContour);

            cv::Point2f rectPoints[4], testPoints[4];
            maskContourRect.points(rectPoints);

            // for test
            //cv::Mat outputImage = cv::Mat::zeros(binaryMap.size(), CV_8UC3);
            //cv::cvtColor(binaryMap, outputImage, cv::COLOR_GRAY2BGR);
            //for (int j = 0; j < 4; j++) 
            //{
            //    cv::line(outputImage, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
            //}
            //cv::imshow("test", outputImage);
            //cv::waitKey(0);
            // for test

            std::sort(rectPoints, rectPoints + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
                return a.y < b.y; });

            if (i == 0 && j == 0 )  // ivs对点重新进行调整
            {
                processPoints(rectPoints);
                // for test
                //cv::Mat dstImageEd = video[0].clone();

                //cv::Point2f testPoints[4];
                //for (int i = 0; i < 4; i++) 
                //{
                //    testPoints[i].x = static_cast<int>(rectPoints[i].x * scaleW);
                //    testPoints[i].y = static_cast<int>(rectPoints[i].y * scaleH);
                //}
                //for (auto& point : testPoints) 
                //{
                //    cv::circle(dstImageEd, point, 2, (0, 0, 255), -1);
                //}
                //cv::imshow("test", dstImageEd);
                //cv::waitKey(0);
                // for test
            }

            for (int i = 0; i < 4; i++)
            {
                testPoints[i].x = static_cast<int>(rectPoints[i].x * scaleW);
                testPoints[i].y = static_cast<int>(rectPoints[i].y * scaleH);
            }

            cv::Point2f upPoint, downPoint;  // 映射回原图
            upPoint.x = (rectPoints[0].x + rectPoints[1].x) * 0.5 * scaleW;
            upPoint.y = (rectPoints[0].y + rectPoints[1].y) * 0.5 * scaleH;
            downPoint.x = (rectPoints[2].x + rectPoints[3].x) * 0.5 * scaleW;
            downPoint.y = (rectPoints[2].y + rectPoints[3].y) * 0.5 * scaleH;

            keypoints.assign({ upPoint, downPoint });
            vKeypoints.push_back(keypoints);
            vRedMasks.push_back(redMask);
            vRectPoints.push_back(std::vector<cv::Point2f>(testPoints, testPoints + 4));
            keypoints.clear();

            float distance = ParamsAssessUtils::calcLineDist(upPoint, downPoint);
            vDistances.push_back(distance);
        }
    }

    //std::vector<std::string> vClassNames = { "IVSd", "LVPWd" };
    std::vector<std::string> vClassNames = { "IVSTd", "LVPWTd" };

    for (int i = 0; i < vClassNames.size(); i++)
    {

        cv::Mat dstImage = video[0].clone();
        std::vector<float> dist = { vDistances[i] };

        cv::Mat blendImage;
        cv::resize(vRedMasks[i], vRedMasks[i], dstImage.size());
        cv::addWeighted(vRedMasks[i], 0.3, dstImage, 0.7, 0.0, blendImage);
        cv::line(blendImage, vKeypoints[i][0], vKeypoints[i][1], cv::Scalar(0, 255, 0), 2);
        std::vector<cv::Point2f> rectPoints = vRectPoints[i];
        for (int j = 0; j < 4; j++) 
        {
            cv::circle(blendImage, rectPoints[j], 2, cv::Scalar(255, 0, 0), -1);
        }
        std::string length = std::to_string(static_cast<int>(m_scaleLength));
        std::string distance = std::to_string(dist[0] / m_scale);
        cv::putText(blendImage, length, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2);
        cv::putText(blendImage, distance, cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2);

        //cv::imshow("test", dstImage);
        //cv::waitKey(0);
        resultPics.emplace(vClassNames[i], blendImage);
        values.emplace(vClassNames[i], dist);

    }
    return 0;
}

int MultiLineAssess::postProcessFourHeatmap(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
    std::vector<std::vector<cv::Point2f>> vKeypoints, vRectPoints;
    std::vector<float> vDistances;
    std::vector<cv::Mat> vRedMasks;

    //cv::Size originalImageSize = video[0].size();
    //cv::Size currImageSize = vMasks[0][0].size();

    //float scaleH = static_cast<float>(originalImageSize.height) / static_cast<float>(currImageSize.height);
    //float scaleW = static_cast<float>(originalImageSize.width) / static_cast<float>(currImageSize.width);


    for (size_t i = 0; i < vMasks.size(); ++i)
    {
        cv::Size originalImageSize;
        if (i == 0) 
        {
            originalImageSize = video[0].size();
        }
        else 
        {
           originalImageSize = (video.back()).size();
        }
        cv::Size currImageSize = vMasks[0][0].size();
        float scaleH = static_cast<float>(originalImageSize.height) / static_cast<float>(currImageSize.height);
        float scaleW = static_cast<float>(originalImageSize.width) / static_cast<float>(currImageSize.width);

        for (size_t j = 0; j < vMasks[i].size(); ++j)
        {
            cv::Mat& heatmap = vMasks[i][j];
            cv::Mat normalizedMap, binaryMap;
            std::vector<cv::Point2f> keypoints;

            //cv::threshold(mat, mat, 0, 255, cv::THRESH_TOZERO);

            cv::normalize(heatmap, normalizedMap, 0, 255, cv::NORM_MINMAX);  // 归一化到0-255
            normalizedMap.convertTo(normalizedMap, CV_8UC1);

            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            int thresh;

            // 查找Mat中的最小值和最大值及其位置
            cv::minMaxLoc(normalizedMap, &minVal, &maxVal, &minLoc, &maxLoc);

            if (maxVal <= 10)
                thresh = static_cast<int>(maxVal);
            else
                thresh = static_cast<int>(0.5 * maxVal);
            cv::threshold(normalizedMap, binaryMap, thresh, 255, cv::THRESH_BINARY);

            // for test
            //cv::imshow("test", binaryMap);
            //cv::waitKey(0);
            // for test

            cv::Mat redMask;
            std::vector<cv::Mat> channels = { cv::Mat::zeros(binaryMap.size(), CV_8UC1),  // Blue channel (0)
                                             cv::Mat::zeros(binaryMap.size(), CV_8UC1),  // Green channel (0)
                                             binaryMap };                                 // Red channel

            cv::merge(channels, redMask);

            std::vector<std::vector<cv::Point>> maskContours;
            cv::findContours(binaryMap, maskContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (maskContours.size() == 0)
            {
                vKeypoints.push_back({ cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f) });
                vDistances.push_back(0.f);
                vRedMasks.push_back(redMask);
                vRectPoints.push_back({ cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f) });
                continue;
            }

            std::vector<cv::Point> maskContour = ImageProcess::findMaxContour(maskContours);
            if (maskContour.empty())
            {
                vKeypoints.push_back({ cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f) });
                vDistances.push_back(0.f);
                vRedMasks.push_back(redMask);
                vRectPoints.push_back({ cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f) });
                continue;
            }
            cv::RotatedRect maskContourRect = cv::minAreaRect(maskContour);

            cv::Point2f rectPoints[4], testPoints[4];
            maskContourRect.points(rectPoints);

            // for test
            //cv::Mat outputImage = cv::Mat::zeros(binaryMap.size(), CV_8UC3);
            //cv::cvtColor(binaryMap, outputImage, cv::COLOR_GRAY2BGR);
            //for (int j = 0; j < 4; j++) 
            //{
            //    cv::line(outputImage, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
            //}
            //cv::imshow("test", outputImage);
            //cv::waitKey(0);
            // for test

            std::sort(rectPoints, rectPoints + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
                return a.y < b.y; });

            //if (i == 0 && (j == 3 || j == 4))  // ivs和pw对点重新进行调整
            //{
            //    processPoints(rectPoints);
                // for test
                //cv::Mat dstImageEd = video[0].clone();

                //cv::Point testPoints[4];
            for (int i = 0; i < 4; i++) 
            {
                testPoints[i].x = static_cast<int>(rectPoints[i].x * scaleW);
                testPoints[i].y = static_cast<int>(rectPoints[i].y * scaleH);
            }
                //for (auto& point : testPoints) 
                //{
                //    cv::circle(dstImageEd, point, 2, (0, 0, 255), -1);
                //}
                //cv::imshow("test", dstImageEd);
                //cv::waitKey(0);
                // for test
            //}

            cv::Point2f upPoint, downPoint;  // 映射回原图
            upPoint.x = (rectPoints[0].x + rectPoints[1].x) * 0.5 * scaleW;
            upPoint.y = (rectPoints[0].y + rectPoints[1].y) * 0.5 * scaleH;
            downPoint.x = (rectPoints[2].x + rectPoints[3].x) * 0.5 * scaleW;
            downPoint.y = (rectPoints[2].y + rectPoints[3].y) * 0.5 * scaleH;

            keypoints.assign({ upPoint, downPoint });
            vKeypoints.push_back(keypoints);
            vRedMasks.push_back(redMask);
            vRectPoints.push_back(std::vector<cv::Point2f>(testPoints, testPoints + 4));
            keypoints.clear();

            float distance = ParamsAssessUtils::calcLineDist(upPoint, downPoint);
            vDistances.push_back(distance);
        }
    }

    //std::vector<std::string> vClassNames = { "Sinus of Valsalvad", "Sinotubular Junction", "Ascending Aorta", "LVIDd", "LAAd" };
    std::vector<std::string> vClassNames = { "ASD", "SJD", "AAD" };
    //std::vector<std::string> vClassNames = { "ASD", "SJD", "AAD", "LVDd", "LAD" };
    for (int i = 0; i < vClassNames.size(); i++)
    {
        if (i <= 3)
        {
            cv::Mat dstImage = video[0].clone();
            std::vector<float> dist = { vDistances[i] };
            cv::Mat blendImage;
            cv::resize(vRedMasks[i], vRedMasks[i], dstImage.size());
            cv::addWeighted(vRedMasks[i], 0.3, dstImage, 0.7, 0.0, blendImage);
            cv::line(blendImage, vKeypoints[i][0], vKeypoints[i][1], cv::Scalar(0, 255, 0), 2);

            std::vector<cv::Point2f> rectPoints = vRectPoints[i];
            for (int j = 0; j < 4; j++)
            {
                cv::circle(blendImage, rectPoints[j], 2 ,  cv::Scalar(255, 0, 0), -1);
            }
            std::string length = std::to_string(static_cast<int>(m_scaleLength));
            std::string distance = std::to_string(dist[0] / m_scale);
            cv::putText(blendImage, length, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2);
            cv::putText(blendImage, distance, cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2);

            //cv::imshow("test", dstImage);
            //cv::waitKey(0);
            resultPics.emplace(vClassNames[i], blendImage);
            values.emplace(vClassNames[i], dist);
        }
        else
        {
            cv::Mat dstImage = video.back().clone();
            std::vector<float> dist = { vDistances[i] };
            cv::Mat blendImage;
            cv::resize(vRedMasks[i], vRedMasks[i], dstImage.size());
            cv::addWeighted(vRedMasks[i], 0.3, dstImage, 0.7, 0.0, blendImage);
            cv::line(blendImage, vKeypoints[i][0], vKeypoints[i][1], cv::Scalar(0, 255, 0), 2);

            std::vector<cv::Point2f> rectPoints = vRectPoints[i];
            for (int j = 0; j < 4; j++)
            {
                cv::circle(blendImage, rectPoints[j], 2, cv::Scalar(255, 0, 0), -1);
            }
            std::string length = std::to_string(static_cast<int>(m_scaleLength));
            std::string distance = std::to_string(dist[0] / m_scale);
            cv::putText(blendImage, length, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2);
            cv::putText(blendImage, distance, cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2);

            //cv::imshow("test", dstImage);
            //cv::waitKey(0);
            resultPics.emplace(vClassNames[i], blendImage);
            values.emplace(vClassNames[i], dist);
        }
    }

    return 0;
}

int MultiLineAssess::postProcessMultiLineMask(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{
    std::vector<std::vector<cv::Point2f>> vKeypoints;
    std::vector<float> vDistances;
    std::vector<cv::Mat> vRedMasks;

    cv::Size originalImageSize = video[0].size();
    cv::Size currImageSize = vMasks[0][0].size();

    float scaleH = static_cast<float>(originalImageSize.height) / static_cast<float>(currImageSize.height);
    float scaleW = static_cast<float>(originalImageSize.width) / static_cast<float>(currImageSize.width);

    std::vector<std::vector<cv::Mat>> vLineMasks(vMasks.begin(), vMasks.begin() + 2);
    std::vector<cv::Mat> vSegMasks(vMasks[2]);

    for (size_t i = 0; i < vLineMasks.size(); ++i)
    {
        for (size_t j = 0; j < vLineMasks[i].size(); ++j)
        {
            cv::Mat& heatmap = vLineMasks[i][j];
            cv::Mat normalizedMap, binaryMap;
            std::vector<cv::Point2f> keypoints;

            //cv::threshold(mat, mat, 0, 255, cv::THRESH_TOZERO);

            cv::normalize(heatmap, normalizedMap, 0, 255, cv::NORM_MINMAX);  // 归一化到0-255
            normalizedMap.convertTo(normalizedMap, CV_8UC1);

            double minVal, maxVal;
            cv::Point minLoc, maxLoc;
            int thresh;

            // 查找Mat中的最小值和最大值及其位置
            cv::minMaxLoc(normalizedMap, &minVal, &maxVal, &minLoc, &maxLoc);

            if (maxVal <= 10)
                thresh = static_cast<int>(maxVal);
            else
                thresh = static_cast<int>(0.5 * maxVal);
            cv::threshold(normalizedMap, binaryMap, thresh, 255, cv::THRESH_BINARY);

            // for test
            //cv::imshow("test", binaryMap);
            //cv::waitKey(0);
            // for test

            cv::Mat redMask;
            std::vector<cv::Mat> channels = { cv::Mat::zeros(binaryMap.size(), CV_8UC1),  // Blue channel (0)
                                             cv::Mat::zeros(binaryMap.size(), CV_8UC1),  // Green channel (0)
                                             binaryMap };                                 // Red channel

            cv::merge(channels, redMask);

            std::vector<std::vector<cv::Point>> maskContours;
            cv::findContours(binaryMap, maskContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            if (maskContours.size() == 0)
            {
                vKeypoints.push_back({ cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f) });
                vDistances.push_back(0.f);
                vRedMasks.push_back(redMask);
                continue;
            }

            std::vector<cv::Point> maskContour = ImageProcess::findMaxContour(maskContours);
            if (maskContour.empty())
            {
                vKeypoints.push_back({ cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f) });
                vDistances.push_back(0.f);
                vRedMasks.push_back(redMask);
                continue;
            }
            cv::RotatedRect maskContourRect = cv::minAreaRect(maskContour);

            cv::Point2f rectPoints[4];
            maskContourRect.points(rectPoints);

            // for test
            //cv::Mat outputImage = cv::Mat::zeros(binaryMap.size(), CV_8UC3);
            //cv::cvtColor(binaryMap, outputImage, cv::COLOR_GRAY2BGR);
            //for (int j = 0; j < 4; j++) 
            //{
            //    cv::line(outputImage, rectPoints[j], rectPoints[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
            //}
            //cv::imshow("test", outputImage);
            //cv::waitKey(0);
            // for test

            std::sort(rectPoints, rectPoints + 4, [](const cv::Point2f& a, const cv::Point2f& b) {
                return a.y < b.y; });

            //if (i == 0 && (j == 3 || j == 4))  // ivs和pw对点重新进行调整
            //{
            //    processPoints(rectPoints);
                // for test
                //cv::Mat dstImageEd = video[0].clone();

                //cv::Point testPoints[4];
                //for (int i = 0; i < 4; i++) 
                //{
                //    testPoints[i].x = static_cast<int>(rectPoints[i].x * scaleW);
                //    testPoints[i].y = static_cast<int>(rectPoints[i].y * scaleH);
                //}
                //for (auto& point : testPoints) 
                //{
                //    cv::circle(dstImageEd, point, 2, (0, 0, 255), -1);
                //}
                //cv::imshow("test", dstImageEd);
                //cv::waitKey(0);
                // for test
            //}

            cv::Point2f upPoint, downPoint;  // 映射回原图
            upPoint.x = (rectPoints[0].x + rectPoints[1].x) * 0.5 * scaleW;
            upPoint.y = (rectPoints[0].y + rectPoints[1].y) * 0.5 * scaleH;
            downPoint.x = (rectPoints[2].x + rectPoints[3].x) * 0.5 * scaleW;
            downPoint.y = (rectPoints[2].y + rectPoints[3].y) * 0.5 * scaleH;

            keypoints.assign({ upPoint, downPoint });
            vKeypoints.push_back(keypoints);
            vRedMasks.push_back(redMask);
            keypoints.clear();

            float distance = ParamsAssessUtils::calcLineDist(upPoint, downPoint);
            vDistances.push_back(distance);
        }
    }

    // 单独处理中心线heatmap
    cv::Mat midLineHeatmap = vLineMasks[0][3];
    cv::Mat midLineHeatmapResized;
    cv::resize(midLineHeatmap, midLineHeatmapResized, originalImageSize);

    cv::Mat normalizedMidLineMap, binaryMidLineMap, skeletonMidLineMap;

    cv::normalize(midLineHeatmapResized, normalizedMidLineMap, 0, 255, cv::NORM_MINMAX);  // 归一化到0-255
    normalizedMidLineMap.convertTo(normalizedMidLineMap, CV_8UC1);

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    int thresh;

    cv::minMaxLoc(normalizedMidLineMap, &minVal, &maxVal, &minLoc, &maxLoc);
    if (maxVal <= 10)
        thresh = static_cast<int>(maxVal);
    else
        thresh = static_cast<int>(0.5 * maxVal);
    cv::threshold(normalizedMidLineMap, binaryMidLineMap, thresh, 255, cv::THRESH_BINARY);
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8UC1);
    cv::erode(binaryMidLineMap, binaryMidLineMap, kernel);
    //cv::ximgproc::thinning(binaryMidLineMap, skeletonMidLineMap);
    ImageProcess::skeletonize(binaryMidLineMap, skeletonMidLineMap);

    //cv::imshow("binary", binaryMidLineMap);
    //cv::imshow("test", skeletonMidLineMap);
    //cv::waitKey(0);

    std::vector<cv::Point> midCurve;
    ImageProcess::getSkelPoints(skeletonMidLineMap, midCurve, 255);  // 得到细化后的中心曲线

    if (midCurve.size() == 0) 
    {
        return 0;
    }
    std::vector<double> midCurveCoefficients;
    ImageProcess::polyfit(midCurve, 3, midCurveCoefficients);

    std::vector<cv::Point2f> lvidPoints = vKeypoints[2];
    if (lvidPoints.size() == 0) 
    {
        lvidPoints = { cv::Point(0, 0), cv::Point(0, 0) };
    }
    cv::Point lvidUpPoint(lvidPoints[0]);
    cv::Point lvidBottomPoint(lvidPoints[1]);
    cv::Point intersection;
    ImageProcess::findClosestPointOnLine(lvidUpPoint, lvidBottomPoint, midCurve, intersection);  // 计算网络预测lvid和中心曲线交点

    std::vector<cv::Point> newLvidEndPoints;
    ImageProcess::getPerpendicularLineEndpoints(midCurveCoefficients, intersection, newLvidEndPoints, 350);  // 基于拟合曲线计算垂线端点

    // 处理分割mask
    cv::Mat segMask = vSegMasks[0];
    cv::Mat segMaskResied;
    cv::resize(segMask, segMaskResied, originalImageSize, 0, 0, 0);  // 注意使用最近邻插值
    std::vector<std::vector<cv::Point>> maskIvsContours, maskPwContours;;

    cv::Mat binaryEdMask = cv::Mat::zeros(originalImageSize, CV_8UC1);
    cv::Mat binaryIvsMask = cv::Mat::zeros(originalImageSize, CV_8UC1);
    cv::Mat binaryPwMask = cv::Mat::zeros(originalImageSize, CV_8UC1);
    binaryEdMask.setTo(255, segMaskResied != 0);
    binaryIvsMask.setTo(255, segMaskResied == 1);
    binaryPwMask.setTo(255, segMaskResied == 2);

    //cv::circle(binaryEdMask, newLvidEndPoints[0], 3, cv::Scalar(122), -1);
    //cv::circle(binaryEdMask, newLvidEndPoints[1], 3, cv::Scalar(122), -1);

    //for (const auto& point : midCurve)
    //{
    //    binaryEdMask.at<uchar>(point) = 255;
    //}
    //cv::imshow("mask", binaryEdMask);
    //cv::waitKey(0);

    cv::findContours(binaryIvsMask, maskIvsContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::findContours(binaryPwMask, maskPwContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    std::vector<cv::Point> ivsIntersections, pwIntersections;
    if (maskIvsContours.size() != 0) {
        ImageProcess::findLineContourIntersections(newLvidEndPoints, maskIvsContours, ivsIntersections);
        ImageProcess::pointSortByY(ivsIntersections);
    }
    else {
        ivsIntersections = { cv::Point(0, 0), cv::Point(0, 0) };
    }
    if (maskPwContours.size() != 0) {
        ImageProcess::findLineContourIntersections(newLvidEndPoints, maskPwContours, pwIntersections);
        ImageProcess::pointSortByY(pwIntersections);
    }
    else {
        pwIntersections = { cv::Point(0, 0), cv::Point(0, 0) };
    }

    std::vector<cv::Point> ladPoints;
    for (const auto& point : vKeypoints.back()) {
        ladPoints.emplace_back(static_cast<int>(point.x), static_cast<int>(point.y));
    }

    std::vector<std::vector<cv::Point>> vLinePoints;
    if (ivsIntersections.size() == 2 && pwIntersections.size() == 2) {
        vLinePoints = { ivsIntersections, pwIntersections, {ivsIntersections.back(), pwIntersections[0]}, ladPoints };
    }
    else{
        vLinePoints = { {cv::Point(0, 0), cv::Point(0, 0)}, {cv::Point(0, 0), cv::Point(0, 0)}, {cv::Point(0, 0), cv::Point(0, 0)}, ladPoints };
    }
    

    // 依据后处理结果在图像上绘制
    std::vector<std::string> vClassNames = { "IVSTd", "LVPWTd" , "LVDd", "LAD" };

    cv::Mat edFrame = video[0].clone();
    cv::Mat redMask, blendEdFrame;
    std::vector<cv::Mat> channels = { cv::Mat::zeros(originalImageSize, CV_8UC1),  // Blue channel (0)
                                 cv::Mat::zeros(originalImageSize, CV_8UC1),  // Green channel (0)
                                 binaryEdMask };                                 // Red channel
    cv::merge(channels, redMask);
    cv::addWeighted(redMask, 0.3, edFrame, 0.7, 0.0, blendEdFrame);
    for (const auto& point : midCurve)
    {
        cv::circle(blendEdFrame, point, 1, cv::Scalar(0, 255, 0));
    }
    for (const auto& point : newLvidEndPoints) 
    {
        cv::circle(blendEdFrame, point, 5, cv::Scalar(255, 0, 0), -1);
    }
    for (int i = 0; i < vClassNames.size() - 1; i++) 
    {
        //cv::line(blendEdFrame, vLinePoints[i][0], vLinePoints[i][1], cv::Scalar(0, 255, 0), 2);
        drawDashedLine(blendEdFrame, vLinePoints[i][0], vLinePoints[i][1], cv::Scalar(255, 255, 255), 1);
        drawCross(blendEdFrame, vLinePoints[i][0], 10, cv::Scalar(255, 255, 255), 3);
        drawCross(blendEdFrame, vLinePoints[i][1], 10, cv::Scalar(255, 255, 255), 3);

        // 在测值线右下侧放置测值量
        float dist = ParamsAssessUtils::calcLineDist(vLinePoints[i]);
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << (dist / m_scale);
        std::string distance = oss.str();
        //std::string distance = std::to_string(dist / m_scale);
        cv::Point textLocation = vLinePoints[i][0];
        textLocation.x += 10;
        textLocation.y += 10;
        cv::putText(blendEdFrame, distance, textLocation, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
    }

    std::map<std::string, std::vector<float>> realValues;
    std::map<std::string, std::vector<float>> tempValues;
    std::map<std::string, cv::Mat> tempResultPics;
    for (int i = 0; i < vClassNames.size(); i++)
    {

        if (i <= 2)
        {
            cv::Mat dstImage = blendEdFrame.clone();
            float dist = ParamsAssessUtils::calcLineDist(vLinePoints[i]);
            std::vector<float> vDist = { dist };

            //cv::line(dstImage, vLinePoints[i][0], vLinePoints[i][1], cv::Scalar(0, 255, 0), 2);

            std::string length = std::to_string(static_cast<int>(m_scaleLength));
            std::string distance = std::to_string(dist / m_scale);
            std::vector<float> fDistance = { 0.f };
            if (m_scale != -1000) 
            {
                fDistance = { dist / m_scale };
            }

            //cv::putText(dstImage, length, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2);
            //cv::putText(dstImage, distance, cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2);
            //cv::imshow("test", dstImage);
            //cv::waitKey(0);
            tempResultPics.emplace(vClassNames[i], dstImage);
            tempValues.emplace(vClassNames[i], vDist);
            realValues.emplace(vClassNames[i], fDistance);
        }
        else
        {
            cv::Mat dstImage = video.back().clone();
            float dist = ParamsAssessUtils::calcLineDist(vLinePoints[i]);
            std::vector<float> vDist = { dist };

            //cv::line(dstImage, vLinePoints[i][0], vLinePoints[i][1], cv::Scalar(0, 255, 0), 2);
            drawDashedLine(dstImage, vLinePoints[i][0], vLinePoints[i][1], cv::Scalar(255, 255, 255), 1);
            drawCross(dstImage, vLinePoints[i][0], 10, cv::Scalar(255, 255, 255), 3);
            drawCross(dstImage, vLinePoints[i][0], 10, cv::Scalar(255, 255, 255), 3);

            std::string length = std::to_string(static_cast<int>(m_scaleLength));

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << (dist / m_scale);
            std::string distance = oss.str();
            //std::string distance = std::to_string(dist / m_scale);

            std::vector<float> fDistance = { 0.f };
            if (m_scale != -1000)
            {
                fDistance = { dist / m_scale };
            }
            //std::vector<float> fDistance = { dist / m_scale };

            cv::Point textLocation = vLinePoints[i][0];
            textLocation.x += 10;
            textLocation.y += 10;

            //cv::putText(dstImage, length, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2);
            cv::putText(dstImage, distance, textLocation, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
            //cv::imshow("test", dstImage);
            //cv::waitKey(0);
            tempResultPics.emplace(vClassNames[i], dstImage);
            tempValues.emplace(vClassNames[i], vDist);
            realValues.emplace(vClassNames[i], fDistance);
        }
    }
    if (!m_values.empty())
        m_values.clear();
    if (!m_resultPics.empty())
        m_resultPics.clear();
    //m_values = realValues;
    //m_resultPics = resultPics;
    parseFinalResults(tempValues, realValues, tempResultPics);

    values = m_values;
    resultPics = m_resultPics;

    return 1;
}

void MultiLineAssess::processPoints(cv::Point2f rectPoints[4])
{
    // 比较第一个和第二个点的y坐标
    if (rectPoints[0].y == rectPoints[1].y) 
    {
        return;
    }

    // 比较第二个和第三个点的x坐标
    if (rectPoints[1].x < rectPoints[2].x) 
    {
        // 交换第二个和第三个点的位置
        cv::Point2f temp = rectPoints[1];
        rectPoints[1] = rectPoints[2];
        rectPoints[2] = temp;
    }
}

void MultiLineAssess::drawCross(cv::Mat& image, const cv::Point& center, int length, const cv::Scalar& color, int thickness)
{
    // 计算叉的两条直线的起点和终点
    cv::Point line1_start(center.x - length / 2, center.y - length / 2);
    cv::Point line1_end(center.x + length / 2, center.y + length / 2);

    cv::Point line2_start(center.x - length / 2, center.y + length / 2);
    cv::Point line2_end(center.x + length / 2, center.y - length / 2);

    // 绘制两条直线
    cv::line(image, line1_start, line1_end, color, thickness);
    cv::line(image, line2_start, line2_end, color, thickness);
}

void MultiLineAssess::drawDashedLine(cv::Mat img, cv::Point p1, cv::Point p2, cv::Scalar color, int thickness)
{
    // https://blog.csdn.net/qq_34801642/article/details/106799817
    float n = 5; //线长度
    float w = p2.x - p1.x, h = p2.y - p1.y;
    float l = sqrtf(w * w + h * h);
    // 矫正线长度，使线个数为奇数
    int m = l / n;
    m = m % 2 ? m : m + 1;
    n = l / m;

    circle(img, p1, 1, color, thickness); // 画起点
    circle(img, p2, 1, color, thickness); // 画终点
    // 画中间点
    if (p1.y == p2.y) //水平线：y = m
    {
        float x1 = std::min(p1.x, p2.x);
        float x2 = std::max(p1.x, p2.x);
        for (float x = x1, n1 = 2 * n; x < x2; x = x + n1)
            line(img, cv::Point2f(x, p1.y), cv::Point2f(x + n, p1.y), color, thickness);
    }
    else if (p1.x == p2.x) //垂直线, x = m
    {
        float y1 = std::min(p1.y, p2.y);
        float y2 = std::max(p1.y, p2.y);
        for (float y = y1, n1 = 2 * n; y < y2; y = y + n1)
            line(img, cv::Point2f(p1.x, y), cv::Point2f(p1.x, y + n), color, thickness);
    }
    else // 倾斜线，与x轴、y轴都不垂直或平行
    {
        // 直线方程的两点式：(y-y1)/(y2-y1)=(x-x1)/(x2-x1) -> y = (y2-y1)*(x-x1)/(x2-x1)+y1
        float n1 = n * abs(w) / l;
        float k = h / w;
        float x1 = std::min(p1.x, p2.x);
        float x2 = std::max(p1.x, p2.x);
        for (float x = x1, n2 = 2 * n1; x < x2; x = x + n2)
        {
            cv::Point p3 = cv::Point2f(x, k * (x - p1.x) + p1.y);
            cv::Point p4 = cv::Point2f(x + n1, k * (x + n1 - p1.x) + p1.y);
            line(img, p3, p4, color, thickness);
        }
    }
}

int MultiLineAssess::parseFinalResults(std::map<std::string, std::vector<float>>& values, std::map<std::string, std::vector<float>>& realValues, std::map<std::string, cv::Mat>& resultPics)
{
    // 检查前三个键是否在范围内
    std::vector<std::string> edKeys = { "IVSTd", "LVPWTd" , "LVDd" };
    std::vector<std::string> esKeys = { "LAD" };
    bool allInRange = true;

    for (const auto& key : edKeys)
    {
        if (realValues.find(key) != realValues.end() && !realValues[key].empty())
        {
            float value = realValues[key][0]; // 取第一个值进行判断
            if (value < m_referRange[key].first || value > m_referRange[key].second)
            {
                allInRange = false;
                break;
            }
        }
        else
        {
            allInRange = false;
            break;
        }
    }

    // 如果前三个值都在范围内
    if (allInRange)
    {
        for (const auto& key : edKeys)
        {
            m_values[key] = realValues[key];
        }
        if (!resultPics.empty())  // 将ivs对应的图像对加入成员变量中
        {
            auto it = resultPics.begin();
            m_resultPics[it->first] = it->second;
        }
    }

    for (const auto& key : esKeys)
    {
        if (realValues.find(key) != realValues.end() && !realValues[key].empty())
        {
            float ladValue = realValues[key][0];
            if (ladValue >= m_referRange[key].first && ladValue <= m_referRange[key].second)
            {
                m_values[key] = realValues[key];
                if (resultPics.find(key) != resultPics.end())
                {
                    m_resultPics[key] = resultPics[key];
                }
            }
        }
    }

    return 1;
}

std::map<std::string, std::vector<float>> MultiLineAssess::getPresentationValues()
{
    return m_values;
}

std::map<std::string, cv::Mat> MultiLineAssess::getPresentationPics()
{
    return m_resultPics;
}

int MultiLineAssess::setScaleInfo(float& scaleLength, float& scale)
{
    m_scaleLength = scaleLength;
    m_scale = scale;
    return 1;
}