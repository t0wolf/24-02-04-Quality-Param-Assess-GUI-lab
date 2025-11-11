#include "ed_es_line_aorta_assess.h"
#include <filesystem>

MultiLineAssess_Aorta::MultiLineAssess_Aorta(std::string& sEnginePath, std::string& sMultiLineMaskPath)
    :m_outpaintPath(sEnginePath)
    , m_multiLineMaskPath_Aorta(sMultiLineMaskPath)
    , m_scale(-10000.0f)
    , m_scaleLength(-1000.0f)
{

    m_outpaint_Inferer = new OutPaintInferer(m_outpaintPath);
    m_multiLineMaskInferer_Aorta = new MultiLineMaskInferer_ed(m_multiLineMaskPath_Aorta);
};

MultiLineAssess_Aorta::~MultiLineAssess_Aorta()
{
	delete m_outpaint_Inferer;
    delete m_multiLineMaskInferer_Aorta;
}



int MultiLineAssess_Aorta::doAortaMultiLineMaskInference(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks)
{
    m_multiLineMaskInferer_Aorta->doInference(video, vMasks);
    return 1;
}
int MultiLineAssess_Aorta::doAortaOutpaintingInference(std::vector<cv::Mat>& video, std::vector<cv::Mat>& outpaint_video)
{
    m_outpaint_Inferer->doInference(video, outpaint_video);
    return 1;
}

int MultiLineAssess_Aorta::AortaMultiLineMaskAssessment(std::vector<cv::Mat>& video, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics)
{   
    std::vector<std::vector<cv::Mat>> vMasks;
    std::vector<cv::Mat> outpaint_video, rotation_video;
    doAortaMultiLineMaskInference(video, vMasks);
    postProcessAortaMultiLineMask(video, vMasks, values, resultPics, rotation_video);
    //std::cout << "values 内容如下：" << std::endl;

    //// 遍历 map 容器中的每个键值对
    //for (const auto& kv : values) {
    //    std::cout << "Key: " << kv.first << ", Values: ";
    //    // 遍历当前键对应的 vector<float>
    //    for (float val : kv.second) {
    //        std::cout << val << " "; // 输出 vector 中的每个浮点数
    //    }
    //    std::cout << std::endl; // 换行
    //}
    //std::vector<std::vector<cv::Mat>> vMasks1;
    //if (rotation_video.size() != 0) {
    //    doAortaOutpaintingInference(rotation_video, outpaint_video);
    //    doAortaMultiLineMaskInference(outpaint_video, vMasks1);
    //    postProcessAortaMultiLineMask(outpaint_video, vMasks1, values, resultPics, rotation_video);
    //}
    //std::cout << "values2 内容如下：" << std::endl;

    //// 遍历 map 容器中的每个键值对
    //for (const auto& kv : values) {
    //    std::cout << "Key: " << kv.first << ", Values: ";
    //    // 遍历当前键对应的 vector<float>
    //    for (float val : kv.second) {
    //        std::cout << val << " "; // 输出 vector 中的每个浮点数
    //    }
    //    std::cout << std::endl; // 换行
    //}

    return 1;
}

//int MultiLineAssess_Aorta::setScaleInfo(float& scaleLength, float& scale)
//{
//    m_scaleLength = scaleLength;
//    m_scale = scale;
//    return 1;
//}


int MultiLineAssess_Aorta::postProcessAortaMultiLineMask(std::vector<cv::Mat>& video, std::vector<std::vector<cv::Mat>>& vMasks1, std::map<std::string, std::vector<float>>& values, std::map<std::string, cv::Mat>& resultPics, std::vector<cv::Mat>& rotation_video)
{
    std::vector<std::vector<cv::Point2f>> vKeypoints;
    std::vector<float> vDistances;
    std::vector<cv::Mat> vRedMasks;

    cv::Size originalImageSize = video[0].size();
    //cv::imshow("test", video[0]);
    //cv::waitKey(0);
    cv::Size currImageSize = vMasks1[0][0].size();

    float scaleH = static_cast<float>(originalImageSize.height) / static_cast<float>(currImageSize.height);
    float scaleW = static_cast<float>(originalImageSize.width) / static_cast<float>(currImageSize.width);

    std::vector<std::vector<cv::Mat>> vLineMasks(vMasks1.begin(), vMasks1.end());
    std::vector<cv::Mat> vSegMasks(vMasks1[1]);

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


            cv::Mat redMask;
            std::vector<cv::Mat> channels = { cv::Mat::zeros(binaryMap.size(), CV_8UC1),  // Blue channel (0)
                                             cv::Mat::zeros(binaryMap.size(), CV_8UC1),  // Green channel (0)
                                             binaryMap };                                 // Red channel

            cv::merge(channels, redMask);

            //cv::imshow("binary_map", normalizedMap);
            //cv::waitKey(0);

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



    std::vector<cv::Point2f> lvidPoints = vKeypoints[2];
    if (lvidPoints.size() == 0)
    {
        lvidPoints = { cv::Point(0, 0), cv::Point(0, 0) };
    }
    cv::Point lvidUpPoint(lvidPoints[0]);
    cv::Point lvidBottomPoint(lvidPoints[1]);
    cv::Point intersection;


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



    std::vector<cv::Point> ladPoints;
    for (const auto& point : vKeypoints.back()) {
        ladPoints.emplace_back(static_cast<int>(point.x), static_cast<int>(point.y));
    }



    // 依据后处理结果在图像上绘制
    std::vector<std::string> vClassNames = { "ASD", "SJD" , "AAD" };

    cv::Mat edFrame = video[0].clone();
    cv::Mat redMask, blendEdFrame;
    std::vector<cv::Mat> channels = { cv::Mat::zeros(originalImageSize, CV_8UC1),  // Blue channel (0)
                                 cv::Mat::zeros(originalImageSize, CV_8UC1),  // Green channel (0)
                                 binaryEdMask };                                 // Red channel
    cv::merge(channels, redMask);
    //cv::imshow("test", redMask);
    //cv::waitKey(0);
    //cv::addWeighted(redMask, 0.3, edFrame, 0.7, 0.0, blendEdFrame);
    blendEdFrame = edFrame;
    //vKeypoints[2][1].y += 10;
    //vKeypoints[2][1].x += 10;
    //vKeypoints[1][1].x += 10;
    for (int i = 0; i < vKeypoints.size() - 1; i++)
    {
        //cv::line(blendEdFrame, vLinePoints[i][0], vLinePoints[i][1], cv::Scalar(0, 255, 0), 2);
        drawDashedLine(blendEdFrame, vKeypoints[i][0], vKeypoints[i][1], cv::Scalar(255, 255, 255), 1);
        drawCross(blendEdFrame, vKeypoints[i][0], 10, cv::Scalar(255, 255, 255), 1);
        drawCross(blendEdFrame, vKeypoints[i][1], 10, cv::Scalar(255, 255, 255), 1);

    }

    std::map<std::string, std::vector<float>> realValues;
    std::map<std::string, std::vector<float>> tempValues;
    std::map<std::string, cv::Mat> tempResultPics;
    if (m_scale > 0 && m_scaleLength > 0) {
        int numLines = static_cast<int>(vKeypoints.size()-1); // 计算线段数量




        // 收集所有线段的方向向量
        std::vector<cv::Point2f> directions;
        bool validAngles = true;
        for (int i = 0; i < numLines; ++i) {
            if (vKeypoints[i].size() >= 2) {  // 确保 vKeypoints[i] 至少有 2 个元素
                cv::Point2f vec = vKeypoints[i][1] - vKeypoints[i][0];
                directions.push_back(vec);
            }
            else {
                // 处理异常情况，例如跳过或记录错误
                validAngles = false;
                break;
            }
        }

        // 角度检查标志
 

        if (numLines == 2) {
            if (directions.size() >= 2) {  // 确保 directions 至少有 2 个元素
                float angle = computeAngle(directions[0], directions[1]);
                if (angle >= 20.0f) validAngles = false;
            }
            else {
                validAngles = false;
            }
        }
        else if (numLines >= 3) {
            if (directions.size() >= 3) {  // 确保 directions 至少有 3 个元素
                for (int i = 0; i < 3; ++i) {
                    for (int j = i + 1; j < 3; ++j) {
                        if (computeAngle(directions[i], directions[j]) >= 20.0f) {
                            validAngles = false;
                            break;
                        }
                    }
                    if (!validAngles) break;
                }
            }
            else {
                validAngles = false;
            }
        }
        else {
            validAngles = false;
        }

        // 只有角度验证通过才进行处理
        if (validAngles) {
            for (int i = 0; i < numLines; ++i) {
                if (i < vClassNames.size()) {  // 确保 vClassNames[i] 存在
                    cv::Mat dstImage = blendEdFrame.clone();
                    cv::Point2f p1 = vKeypoints[i][0];
                    cv::Point2f p2 = vKeypoints[i][1];

                    // 绘制线段
                    cv::line(dstImage, p1, p2, cv::Scalar(0, 255, 0), 2);

                    // 计算实际距离
                    float pixelDist = cv::norm(p2 - p1);
                    float realDist = pixelDist / m_scale;

                    //// 添加标注信息
                    //std::string scaleText = "Scale: " + std::to_string(static_cast<int>(m_scaleLength));
                    //std::string distText = "Distance: " + std::to_string(realDist);

                    //cv::putText(dstImage, scaleText, cv::Point(50, 50),
                    //    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
                    //cv::putText(dstImage, distText, cv::Point(50, 80),
                    //    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
                    std::string length = std::to_string(static_cast<int>(m_scaleLength));

                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << (pixelDist / m_scale);
                    std::string distance = oss.str();
                    //std::string distance = std::to_string(dist / m_scale);

                    std::vector<float> fDistance = { 0.f };
                    //if (m_scale != -1000)
                    //{
                    //    fDistance = { dist / m_scale };
                    //}
                    //std::vector<float> fDistance = { dist / m_scale };
                    for (int j = 0; j < numLines; ++j) {
                        cv::Point textLocation = vKeypoints[j][0];
                        textLocation.x += 10;
                        textLocation.y += 10;
                        cv::Point2f p1 = vKeypoints[j][0];
                        cv::Point2f p2 = vKeypoints[j][1];

                        // 绘制线段
                        //cv::line(dstImage, p1, p2, cv::Scalar(0, 255, 0), 2);

                        // 计算实际距离
                        float pixelDist = cv::norm(p2 - p1);
                        float realDist = pixelDist / m_scale;

                        //// 添加标注信息
                        //std::string scaleText = "Scale: " + std::to_string(static_cast<int>(m_scaleLength));
                        //std::string distText = "Distance: " + std::to_string(realDist);

                        //cv::putText(dstImage, scaleText, cv::Point(50, 50),
                        //    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
                        //cv::putText(dstImage, distText, cv::Point(50, 80),
                        //    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
                        std::string length = std::to_string(static_cast<int>(m_scaleLength));

                        std::ostringstream oss;
                        oss << std::fixed << std::setprecision(2) << (pixelDist / m_scale);
                        std::string distance = oss.str();
                        //cv::putText(dstImage, length, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2);
                        cv::putText(dstImage, distance, textLocation, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
                    }

                    // 存储结果
                    //cv::imshow("test", dstImage);
                    //cv::waitKey(0);
                    tempResultPics.emplace(vClassNames[i], dstImage);
                    resultPics.emplace(vClassNames[i], dstImage);
                    tempValues.emplace(vClassNames[i], std::vector<float>{pixelDist});
                    realValues.emplace(vClassNames[i], std::vector<float>{realDist});
                }
                else {
                    // 处理异常情况，例如跳过或记录错误
                }
            
        }
      }

        //for (int i = 0; i < vKeypoints.size() - 1; i++)
        //{

        //    if (i <= 2)
        //    {
        //        cv::Mat dstImage = blendEdFrame.clone();
        //        cv::Point2f p1 = vKeypoints[i][0];
        //        cv::Point2f p2 = vKeypoints[i][1];

        //        cv::Point2f v = p2 - p1;
        //        float dist = sqrt(v.x * v.x + v.y * v.y);

        //        std::vector<float> vDist = { dist };

        //        cv::line(dstImage, vKeypoints[i][0], vKeypoints[i][1], cv::Scalar(0, 255, 0), 2);

        //        std::string length = std::to_string(static_cast<int>(m_scaleLength));
        //        std::string distance = std::to_string(dist / m_scale);
        //        std::vector<float> fDistance = { dist / m_scale };
        //        //if (dist < 5 && dist > 1)
        //        //{

        //        cv::putText(dstImage, length, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2);
        //        cv::putText(dstImage, distance, cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2);
        //        //cv::imshow("test", dstImage);
        //        //cv::waitKey(0);
        //        tempResultPics.emplace(vClassNames[i], dstImage);
        //        tempValues.emplace(vClassNames[i], vDist);
        //        realValues.emplace(vClassNames[i], fDistance);
        //        resultPics.emplace(vClassNames[i], dstImage);

        //        //}

        //    }

        //}
    }
    parseFinalResults(tempValues, realValues, tempResultPics);

   


    // 检查前三个键是否在范围内
    std::vector<std::string> edKeys = { "ASD", "SJD" , "AAD" };
    //std::vector<std::string> esKeys = { "LAD" };
    //bool allInRange = true;

    for (const auto& key : edKeys)
    {

        if (m_values.find(key) != m_values.end() && !m_values[key].empty()) {
            values.emplace(key, std::vector<float>{m_values[key][0]});
          /// resultPics.insert(m_resultPics.begin(),m_resultPics.end());
        }
        //resultPics.emplace(key,m_resultPics[0]);
    }

    if (!m_values["AAD"].empty())
        values.emplace("AoD", std::vector<float>{m_values["AAD"][0]});
    //else
    //    values.emplace("AoD", std::vector<float>{-1.0f});
    //values = m_values;
    //    // 只对 key = "AAD" 进行修改
    //sresultPics = m_resultPics;
    return 1;
}


int MultiLineAssess_Aorta::setScaleInfo(float& scaleLength, float& scale)
{
    m_scaleLength = scaleLength;
    m_scale = scale;
    return 1;
}

void MultiLineAssess_Aorta::drawCross(cv::Mat& image, const cv::Point& center, int length, const cv::Scalar& color, int thickness)
{
    // 计算叉的两条直线的起点和终点
    cv::Point line1_start(center.x - length / 4, center.y - length / 4);
    cv::Point line1_end(center.x + length / 4, center.y + length / 4);

    cv::Point line2_start(center.x - length / 4, center.y + length / 4);
    cv::Point line2_end(center.x + length / 4, center.y - length / 4);

    // 绘制两条直线
    cv::line(image, line1_start, line1_end, color, thickness);
    cv::line(image, line2_start, line2_end, color, thickness);
}

void MultiLineAssess_Aorta::drawDashedLine(cv::Mat img, cv::Point p1, cv::Point p2, cv::Scalar color, int thickness)
{

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

void MultiLineAssess_Aorta::processPoints(cv::Point2f rectPoints[4])
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

int MultiLineAssess_Aorta::parseFinalResults(std::map<std::string, std::vector<float>>& values, std::map<std::string, std::vector<float>>& realValues, std::map<std::string, cv::Mat>& resultPics)
{
    // 检查前三个键是否在范围内
    std::vector<std::string> edKeys = { "ASD", "SJD"  };
    std::vector<std::string> esKeys = { "AAD" };
    bool allInRange = true;

    for (const auto& key : edKeys)
    {
        if (realValues.find(key) != realValues.end() && !realValues[key].empty())
        {
            float value = realValues[key][0]; // 取第一个值进行判断
            float value2 = realValues["AAD"][0];
            if (value > value2+0.3  ) {
                allInRange = false;
                break;
            }
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
    allInRange = true;

    for (const auto& key : esKeys)
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
        for (const auto& key : esKeys)
        {
            m_values[key] = realValues[key];
        }
        if (!resultPics.empty())  // 将ivs对应的图像对加入成员变量中
        {
            auto it = resultPics.begin();
            m_resultPics[it->first] = it->second;
        }
    }
    //for (const auto& key : esKeys)
    //{
    //    if (realValues.find(key) != realValues.end() && !realValues[key].empty())
    //    {
    //        float ladValue = realValues[key][0];
    //        if (ladValue >= m_referRange[key].first && ladValue <= m_referRange[key].second)
    //        {
    //            m_values[key] = values[key];
    //            if (resultPics.find(key) != resultPics.end())
    //            {
    //                m_resultPics[key] = resultPics[key];
    //            }
    //        }
    //    }
    //}

    return 1;
}


// 计算两条线之间的夹角（以度为单位）
//float MultiLineAssess_Aorta::calculateAngle(const cv::Point2f& v1, const cv::Point2f& v2) {
//    float dotProduct = v1.x * v2.x + v1.y * v2.y;
//    float magnitude1 = sqrt(v1.x * v1.x + v1.y * v1.y);
//    float magnitude2 = sqrt(v2.x * v2.x + v2.y * v2.y);
//    float cosTheta = dotProduct / (magnitude1 * magnitude2);
//    float angle = acos(cosTheta) * 180.0 / CV_PI; // 转换为角度
//    return angle;
//}

// 辅助函数：计算两个向量之间的锐角夹角（单位：度）
float MultiLineAssess_Aorta::computeAngle(const cv::Point2f& v1, const cv::Point2f& v2) {
    float dot = v1.x * v2.x + v1.y * v2.y;
    float norm1 = sqrt(v1.x * v1.x + v1.y * v1.y);
    float norm2 = sqrt(v2.x * v2.x + v2.y * v2.y);

    if (norm1 == 0 || norm2 == 0) return 0.0f;

    float cosTheta = dot / (norm1 * norm2);
    cosTheta = std::max(-1.0f, std::min(1.0f, cosTheta)); // 确保数值有效
    float angle = acos(cosTheta) * 180.0 / CV_PI;

    return angle <= 90.0f ? angle : 180.0f - angle; // 返回锐角
}