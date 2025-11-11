#include "assess_utils.h"

std::vector<float> ParamsAssessUtils::linspace(float start, float end, int num)
{
    std::vector<float> linspaced;
    if (num == 0) {
        return linspaced;
    }
    if (num == 1) {
        linspaced.push_back(start);
        return linspaced;
    }

    double delta = (end - start) / (num - 1);

    for (int i = 0; i < num - 1; ++i) {
        linspaced.push_back(start + delta * i);
    }
    // Ensure that end is exactly included
    linspaced.push_back(end);

    return linspaced;
}

int ParamsAssessUtils::calcLinesDistance(std::vector<std::vector<cv::Point>>& vPoints, std::vector<float>& vDists)
{
    for (int i = 0; i < vPoints.size(); ++i)
    {
        std::vector<cv::Point> point = vPoints[i];
        float fDist = calcLineDist(point);
        vDists.push_back(fDist);
    }

    return 1;
}

int ParamsAssessUtils::removeAbnormalValues(std::vector<float>& vValues)
{
    if (vValues.size() < 3)
        return 0;

    auto it = vValues.begin() + 1;
    while (it != vValues.end() - 1)
    {
        if (*it < *(it - 1) - 10.0f && *it < *(it + 1) - 10.0f)
        {
            it = vValues.erase(it);
        }
        else
        {
            ++it;
        }
    }

    return 1;
}

int ParamsAssessUtils::removeAbnormalInterPoints(std::vector<std::vector<cv::Point>>& vInterPoints)
{
    if (vInterPoints.size() < 3)
        return 0;

    auto it = vInterPoints.begin() + 1;
    while (it != vInterPoints.end() - 1)
    {
        std::vector<cv::Point> currPoints, prevPoints, nextPoints;
        float currDist, prevDist, nextDist;

        currPoints = *it;
        prevPoints = *(it - 1);
        nextPoints = *(it + 1);

        currDist = ParamsAssessUtils::calcLineDist(currPoints);
        prevDist = ParamsAssessUtils::calcLineDist(prevPoints);
        nextDist = ParamsAssessUtils::calcLineDist(nextPoints);

        if (currDist < prevDist - 10.0f && currDist < nextDist - 10.0f)
        {
            it = vInterPoints.erase(it);
        }
        else
        {
            ++it;
        }
    }

    return 1;
}

std::vector<double> ParamsAssessUtils::findLocalMaximum(std::vector<double> Vec)   // 找到极大值
{
    std::vector<double> PeakValues;
    for (size_t i = 1; i < Vec.size() - 1; i++)
    {
        float Current = Vec[i];
        float Left = Vec[i - 1];
        float Right = Vec[i + 1];
        if (Left < Current && Current >= Right)
        {
            PeakValues.push_back(Current);
        }
    }

    return PeakValues;
}

std::vector<std::pair<double, size_t>> ParamsAssessUtils::findLocalMaximumPair(std::vector<double> Vec)   // 找到极大值
{
    std::vector<std::pair<double, size_t>> PeakValues;
    for (size_t i = 1; i < Vec.size() - 1; i++)
    {
        float Current = Vec[i];
        float Left = Vec[i - 1];
        float Right = Vec[i + 1];
        if (Left < Current && Current >= Right)
        {
            PeakValues.push_back(std::make_pair(Current, i));
        }
    }

    return PeakValues;
}

std::vector<double> ParamsAssessUtils::findLocalMinimum(std::vector<double> Vec)   // 找到极小值
{
    std::vector<double> PeakValues;
    for (size_t i = 1; i < Vec.size() - 1; i++)
    {
        float Current = Vec[i];
        float Left = Vec[i - 1];
        float Right = Vec[i + 1];
        if (Left > Current && Current <= Right)
        {
            PeakValues.push_back(Current);
        }
    }

    return PeakValues;
}

std::vector<size_t> ParamsAssessUtils::findLocalMinimumIdx(std::vector<std::pair<double, size_t>> Vec)   // 找到极小值
{
    std::vector<size_t> PeakValues;
    for (size_t i = 1; i < Vec.size() - 1; i++)
    {
        float Current = Vec[i].first;
        float Left = Vec[i - 1].first;
        float Right = Vec[i + 1].first;
        if (Left > Current && Current <= Right)
        {
            PeakValues.push_back(Vec[i].second);
        }
    }

    return PeakValues;
}

std::vector<float> ParamsAssessUtils::gradientOneDimension(std::vector<float> vValues)   // 求一组数的梯度
{
    std::vector<float> Gradient(vValues.size());

    for (size_t i = 1; i < vValues.size() - 1; i++)
    {
        Gradient[i] = (vValues[i + 1] - vValues[i - 1]) / 2.0f;
    }
    Gradient[0] = (vValues[1] - vValues[0]) / 1.0f;
    Gradient[vValues.size() - 1] = (vValues[vValues.size() - 1] - vValues[vValues.size() - 2]) / 1.0f;

    return Gradient;
}

std::vector<float> ParamsAssessUtils::gradientOneDimension(std::vector<double> vValues)   // 求一组数的梯度
{
    if (vValues.size() <= 2)
    {
        std::vector<float> gradient;
        for (auto value : vValues)
        {
            gradient.push_back(static_cast<float>(value));
        }
        return gradient;
    }

    std::vector<float> Gradient(vValues.size());

    for (size_t i = 1; i < vValues.size() - 1; i++)
    {
        Gradient[i] = (vValues[i + 1] - vValues[i - 1]) / 2.0f;
    }
    Gradient[0] = (vValues[1] - vValues[0]) / 1.0f;
    Gradient[vValues.size() - 1] = (vValues[vValues.size() - 1] - vValues[vValues.size() - 2]) / 1.0f;

    return Gradient;
}

std::vector<size_t> ParamsAssessUtils::findIndices(const std::vector<float>& array, float threshold)  // 找到小于某一阈值的数的索引
{
    std::vector<size_t> Indices;

    for (size_t i = 0; i < array.size(); i++)
    {
        if (std::abs(array[i]) < threshold)
        {
            Indices.push_back(i);
        }
    }

    return Indices;
}

float ParamsAssessUtils::sign(float value)
{
    if (value > 0.)
    {
        return 1.0f;
    }
    else if (value == 0.)
    {
        return 0.0f;
    }
    else
    {
        return -1.0f;
    }
}

std::vector<int> ParamsAssessUtils::findPeaks(std::vector<int> x, std::vector<double> plateauSize, std::vector<double> height, std::vector<double> threshold, int distance, std::vector<double> prominence, int wlen, std::vector<double> width, double relHeight)  // https://github.com/johnhillross/find_peaks-CPP/tree/main
{
    std::vector<int> peaks;

    for (int i = 1; i < x.size() - 1; i++)
    {
        if (x[i - 1] < x[i]) {
            int iahead = i + 1;
            while (iahead < x.size() - 1 && x[iahead] == x[i])
            {
                iahead++;
            }
            if (x[iahead] < x[i])
            {
                bool peakflag = true;
                // Evaluate plateau size
                if (plateauSize.size() == 2)
                {
                    int currentPlateauSize = iahead - i;
                    if (currentPlateauSize < plateauSize[0] || currentPlateauSize > plateauSize[1])
                    {
                        peakflag = false;
                    }
                }

                // Evaluate height condition
                if (height.size() == 2)
                {
                    int currentPeakIndex = (i + iahead - 1) / 2;

                    if (x[currentPeakIndex] < height[0] || x[currentPeakIndex] > height[1]) {

                        peakflag = false;
                    }
                }

                // Evaluate threshold condition
                if (threshold.size() == 2)
                {
                    int currentPeakIndex = (i + iahead - 1) / 2;

                    if (std::min(x[currentPeakIndex] - x[currentPeakIndex - 1], x[currentPeakIndex] - x[currentPeakIndex + 1]) < threshold[0] || std::max(x[currentPeakIndex] - x[currentPeakIndex - 1], x[currentPeakIndex] - x[currentPeakIndex + 1]) > threshold[1]) {

                        peakflag = false;
                    }
                }

                if (peakflag)
                {
                    peaks.push_back((i + iahead - 1) / 2);
                }

                i = iahead;
            }
        }
    }

    // Evaluate distance condition
    if (distance > 0) {

        std::vector<bool> eraseIndex(peaks.size(), false);
        std::vector<int> sortPeaks = peaks;
        std::sort(sortPeaks.begin(), sortPeaks.end(), [&x](int pos1, int pos2) {return (x[pos1] > x[pos2]); });	//sort peaks by the value of x[peaks]

        for (int i = 0; i < sortPeaks.size(); i++)
        {
            int j = find(peaks.begin(), peaks.end(), sortPeaks[i]) - peaks.begin();

            if (eraseIndex[j])
            {
                continue;
            }

            int k = j - 1;

            while (k >= 0 && peaks[j] - peaks[k] < distance)
            {
                //int test = peaks[j] - peaks[k];
                eraseIndex[k] = true;
                k--;
            }

            k = j + 1;

            while (k < peaks.size() && peaks[k] - peaks[j] < distance)
            {
                //int testR = peaks[k] - peaks[j];
                eraseIndex[k] = true;
                k++;
            }
        }

        int eraseCount = 0;

        for (int i = 0; i < eraseIndex.size(); i++)
        {
            if (eraseIndex[i])
            {
                peaks.erase(peaks.begin() + (i - eraseCount));
                eraseCount++;
            }
        }
    }

    // Evaluate prominence condition, wlen must be >= 2
    if (prominence.size() == 2 || width.size() == 2)
    {
        std::vector<int> copyPeaks = peaks;
        std::vector<double> prominences;
        std::vector<int> leftBases;
        std::vector<int> rightBases;
        int eraseCount = 0;

        for (int i = 0; i < copyPeaks.size(); i++)
        {
            int imin = 0;
            int imax = x.size() - 1;
            int peak = copyPeaks[i];
            double leftMin = x[peak];
            double rightMin = x[peak];
            int leftIndex = peak;
            int rightIndex = peak;
            int j;
            double currentProminence;

            if (wlen >= 2)
            {
                imin = std::max(peak - wlen / 2, imin);
                imax = std::min(peak + wlen / 2, imax);
            }

            j = peak;

            while (j >= imin && x[j] <= x[peak])
            {
                if (x[j] < leftMin)
                {
                    leftMin = x[j];
                    leftIndex = j;
                }
                j--;
            }

            j = peak;

            while (j <= imax && x[j] <= x[peak])
            {
                if (x[j] < rightMin)
                {
                    rightMin = x[j];
                    rightIndex = j;
                }
                j++;
            }

            currentProminence = x[peak] - std::max(leftMin, rightMin);

            if (prominence.size() == 2)
            {
                if (currentProminence >= prominence[0] && currentProminence <= prominence[1])
                {
                    prominences.push_back(currentProminence);
                    leftBases.push_back(leftIndex);
                    rightBases.push_back(rightIndex);
                }
                else
                {
                    peaks.erase(peaks.begin() + (i - eraseCount));
                    eraseCount++;
                }
            }
            else
            {
                prominences.push_back(currentProminence);
                leftBases.push_back(leftIndex);
                rightBases.push_back(rightIndex);
            }
        }

        // Evaluate width condition
        if (width.size() == 2)
        {

            copyPeaks = peaks;
            eraseCount = 0;

            for (int i = 0; i < copyPeaks.size(); i++) {

                int peak = copyPeaks[i];
                int imin = leftBases[i];
                int imax = rightBases[i];
                double height = x[peak] - prominences[i] * relHeight;
                int j;
                double leftIp;
                double rightIp;
                double currentWidth;

                j = peak;

                while (j > imin && x[j] > height)
                {
                    j--;
                }

                if (x[j] < height)
                {
                    leftIp = j + (height - x[j]) / (x[j + 1] - x[j]);
                }

                j = peak;

                while (j < imax && x[j] > height)
                {
                    j++;
                }

                if (x[j] < height)
                {
                    rightIp = j - (height - x[j]) / (x[j - 1] - x[j]);
                }
                currentWidth = rightIp - leftIp;

                if (currentWidth < width[0] || currentWidth > width[1])
                {
                    peaks.erase(peaks.begin() + i - eraseCount);
                    eraseCount++;
                }
            }
        }
    }

    return peaks;
}



//std::vector<ParamsAssessUtils::Peak> ParamsAssessUtils::findPeaks(const std::vector<double>& x, double height, double threshold, int distance, double prominence, double width, double rel_height, int plateau_size)  // realized by chatgpt3.5
//{
//	std::vector<ParamsAssessUtils::Peak> peaks;
//	const int size = x.size();
//	double prev = x[0];
//
//	for (int i = 1; i < size - 1; ++i) {
//		if (x[i] > prev && x[i] > x[i + 1]) {
//			bool is_peak = true;
//
//			if (height > 0.0 && x[i] < height)
//				is_peak = false;
//
//			if (threshold > 0.0 && (x[i] - x[i - 1]) < threshold)
//				is_peak = false;
//
//			if (distance > 1) {
//				for (int j = 1; j <= distance; ++j) {
//					if (i - j < 0 || i + j >= size || x[i] <= x[i - j] || x[i] <= x[i + j]) {
//						is_peak = false;
//						break;
//					}
//				}
//			}
//
//			if (prominence > 0.0) {
//				double left_min = x[i];
//				double right_min = x[i];
//
//				for (int j = 1; j <= distance; ++j) {
//					double prev_left = x[i - j];
//					double prev_right = x[i + j];
//
//					if (prev_left < left_min)
//						left_min = prev_left;
//					if (prev_right < right_min)
//						right_min = prev_right;
//				}
//
//				double current_prominence = x[i] - std::min(left_min, right_min);
//
//				if (current_prominence < prominence)
//					is_peak = false;
//			}
//
//			if (is_peak) {
//				peaks.push_back({ i, x[i] });
//			}
//		}
//
//		prev = x[i];
//	}
//
//	return peaks;
//}

std::vector<ParamsAssessUtils::PeakIdx> ParamsAssessUtils::findPeakIdxs(const std::vector<int>& x, double height, double threshold, int distance, double prominence, double width, double rel_height, int plateau_size)
{
    std::vector<ParamsAssessUtils::PeakIdx> peaks;
    const int size = x.size();
    int prev = x[0];

    for (int i = 1; i < size - 1; ++i)
    {
        if (x[i] > prev && x[i] > x[i + 1])
        {
            bool is_peak = true;

            if (height > 0.0 && x[i] < height)
                is_peak = false;

            if (threshold > 0.0 && (x[i] - x[i - 1]) < threshold)
                is_peak = false;

            if (is_peak)
            {
                peaks.push_back({ i, x[i] });
            }
        }
        prev = x[i];
    }

    if (peaks.size() > 0)
    {
        if (distance > 1)
        {
            int eraseCount = 0;
            for (int i = 0; i < peaks.size() - 1; i++)
            {
                int currentPeakIdx = peaks[i].index;
                int rightPeakIdx = peaks[i + 1].index;
                if (rightPeakIdx - currentPeakIdx < distance)
                {
                    int smallIdx = peaks[i].height < peaks[i + 1].height ? i : i + 1;

                }
            }
        }
        if (prominence > 0.0)
        {

        }
    }

    /*if (distance > 1) {
                for (int j = 1; j <= distance; ++j) {
                    if (i - j < 0 || i + j >= size || x[i] <= x[i - j] || x[i] <= x[i + j]) {
                        is_peak = false;
                        break;
                    }
                }
            }

            if (prominence > 0.0) {
                int left_min = x[i];
                int right_min = x[i];

                for (int j = 1; j <= distance; ++j) {
                    if (i - j >= 0 && i + j <= size - 1)
                    {
                        int prev_left = x[i - j];
                        int prev_right = x[i + j];

                        if (prev_left < left_min)
                            left_min = prev_left;
                        if (prev_right < right_min)
                            right_min = prev_right;
                    }
                    else
                    {
                        break;
                    }
                }

                int current_prominence = x[i] - std::min(left_min, right_min);

                if (current_prominence < prominence)
                    is_peak = false;
            }*/

    return peaks;
}


double ParamsAssessUtils::calculateStdDev(const std::vector<int>& data, double mean)
{
    double sq_sum = std::inner_product(data.begin(), data.end(), data.begin(), 0.0);
    double variance = sq_sum / data.size() - mean * mean;
    return std::sqrt(variance);
}

double ParamsAssessUtils::calculateMean(const std::vector<int>& data)
{
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

// 筛除偏离数据
std::vector<int> ParamsAssessUtils::filterOutliers(const std::vector<int>& data, double threshold, int& outlierCounter) {
    double mean = calculateMean(data);
    double stddev = calculateStdDev(data, mean);

    std::vector<int> filteredData;
    outlierCounter = 0;

    for (int value : data) {
        if (std::abs(value - mean) <= threshold * stddev) {
            filteredData.push_back(value);
        }
        else {
            ++outlierCounter;
        }
    }

    return filteredData;
}

int ParamsAssessUtils::findMedium(std::vector<int>& inputData)
{
    if (inputData.empty()) 
    {
        return 0;
    }

    // Create a copy of the vector to sort
    std::vector<int> sortedNums = inputData;
    std::sort(sortedNums.begin(), sortedNums.end());

    int n = sortedNums.size();
    // Determine the median
    if (n % 2 != 0) 
    {
        // If the number of elements is odd, return the middle element
        return sortedNums[n / 2];
    }
    else 
    {
        // If the number of elements is even, return the average of the two middle elements
        return (sortedNums[(n - 1) / 2] + sortedNums[n / 2]) / 2.0;
    }
}

int ParamsAssessUtils::generateFinalResult(const std::string valueName, 
    std::vector<int>& originValues, 
    std::map<std::string, std::vector<float>>& values,
    cv::Mat& showingPics,
    std::map<std::string, cv::Mat>& resultPics)
{
    int pointsSize = originValues.size();
    int nOutlier = 0;
    originValues = ParamsAssessUtils::filterOutliers(originValues, 1.5, nOutlier);

    int currThresh = static_cast<int>(std::floor(static_cast<float>(pointsSize) * 0.4f));
    if (nOutlier > currThresh)
        return 0;

    std::vector<float> fOriginValues;
    for (auto& value : originValues)
        fOriginValues.push_back(static_cast<float>(value));

    values.insert({ valueName, fOriginValues });
    resultPics.insert({ valueName, showingPics });

    return 1;
}

int ParamsAssessUtils::parseKeyframes(std::vector<int>& keyframeIdx, std::vector<cv::Mat>& inputVideo, std::vector<cv::Mat>& outputVideo, int length)
{
    QtLogger::instance().logMessage(QString("[I] Parsing keyframe ..."));
    // 参数检查
    if (inputVideo.empty() || length <= 0) {
        return 0; // 无效输入
    }

    if (keyframeIdx.empty())
        return 0;

    size_t videoSize = inputVideo.size();

    // 帮助函数：帧补齐
    auto padFrames = [&](int startIndex, int endIndex, int targetLength) {
        if (startIndex < 0 || startIndex > inputVideo.size() - 1)
        {
            QtLogger::instance().logMessage(QString("[E] startIndex out of range, got %1, expected %2").arg(startIndex).arg(inputVideo.size() - 1));
            startIndex = std::min(static_cast<size_t>(startIndex), inputVideo.size() - 1);
        }
        if (endIndex < 0 || endIndex > inputVideo.size() - 1)
        {
            QtLogger::instance().logMessage(QString("[E] endIndex out of range, got %1, expected %2").arg(endIndex).arg(inputVideo.size() - 1));
            endIndex = std::min(static_cast<size_t>(endIndex), inputVideo.size() - 1);
        }

        for (int i = startIndex; i <= endIndex; i++) {
            outputVideo.push_back(inputVideo[i]);
        }
        while (outputVideo.size() < static_cast<size_t>(targetLength)) {
            outputVideo.push_back(inputVideo[endIndex].clone());
        }
        };

    // 帮助函数：按等间隔采样帧
    auto sampleFrames = [&](int startIndex, int endIndex, int targetLength) {
        if (startIndex < 0 || startIndex > inputVideo.size() - 1)
        {
            QtLogger::instance().logMessage(QString("[E] startIndex out of range, got %1, expected %2").arg(startIndex).arg(inputVideo.size() - 1));
        }
        if (endIndex < 0 || endIndex > inputVideo.size() - 1)
        {
            QtLogger::instance().logMessage(QString("[E] endIndex out of range, got %1, expected %2").arg(endIndex).arg(inputVideo.size() - 1));
        }

        double interval = static_cast<double>(endIndex - startIndex) / (targetLength - 1);
        for (int i = 0; i < targetLength; ++i) {
            size_t index = static_cast<size_t>(std::round(startIndex + i * interval));
            index = std::min(index, inputVideo.size() - 1);
            if (index < 0 || index > inputVideo.size() - 1)
            {
                QtLogger::instance().logMessage(QString("[E] sampleIndex out of range, got %1, expected %2").arg(index).arg(inputVideo.size() - 1));
            }
            outputVideo.push_back(inputVideo[index]);
        }
        };

    // 帮助函数：检查并调整帧顺序（确保 ED 在开头，ES 在结尾）
    auto ensureFrameOrder = [&](const std::pair<int, int>& lastPair) {
        // 如果索引对的 first 是负数（ES），second 是正数（ED），则需要反序
        if (lastPair.first < 0 && lastPair.second >= 0) {
            std::reverse(outputVideo.begin(), outputVideo.end());
            //for (auto& mat : outputVideo)
            //{
            //    cv::imshow("after", mat);
            //    cv::waitKey(0);
            //}
        }
        };

    // 如果只有一个索引
    if (keyframeIdx.size() == 1) 
    {
        QtLogger::instance().logMessage(QString("[I] Keyframe only 1 index."));
        int idx = keyframeIdx[0];
        if (abs(idx) > static_cast<int>(videoSize)) {
            padFrames(videoSize - length, videoSize - 1, length);
        }
        QtLogger::instance().logMessage(QString("[I] Keyframe 1 index parsing done."));
        return 1;
    }

    // 如果有多个索引
    QtLogger::instance().logMessage(QString("[I] Keyframe multiple indices."));
    std::vector<std::pair<int, int>> selectedIdx;
    std::sort(keyframeIdx.begin(), keyframeIdx.end(), [](int a, int b) {
        return std::abs(a) < std::abs(b);
        });
    for (size_t i = 0; i < keyframeIdx.size() - 1; ++i) {
        //if (keyframeIdx[i] >= 0 && keyframeIdx[i + 1] < 0) {
        //    selectedIdx.emplace_back(keyframeIdx[i], keyframeIdx[i + 1]);
        //}
        selectedIdx.emplace_back(keyframeIdx[i], keyframeIdx[i + 1]);
    }

    if (!selectedIdx.empty()) 
    {
        // 选择最后一组 (ED, ES) 对
        std::pair<int, int> pairFinalSelectIdx{-10000, -10000};
        for (int i = 0; i < selectedIdx.size(); ++i)
        {
            std::pair<int, int> pairIdx = selectedIdx[i];
            if (std::abs(pairIdx.first - pairIdx.second) < 5)
            {
                continue;
            }
            pairFinalSelectIdx = pairIdx;
        }
        if (pairFinalSelectIdx.first == -10000 || pairFinalSelectIdx.second == -10000)
        {
            QtLogger::instance().logMessage(QString("[E] ED ES Distance Close: %1 %2").arg(selectedIdx.back().first).arg(selectedIdx.back().second));
            return 0;
        }

        //auto lastPair = selectedIdx.back();
        //int startIdx = abs(lastPair.first);
        //int endIdx = abs(lastPair.second);
        int startIdx = abs(pairFinalSelectIdx.first);
        int endIdx = abs(pairFinalSelectIdx.second);

        if (endIdx - startIdx + 1 >= length) {
            sampleFrames(startIdx, endIdx, length);
        }
        else {
            padFrames(startIdx, endIdx, length);
        }

        //int edIndex = lastPair.first;
        //int esIndex = abs(lastPair.second); // ES 索引绝对值

        //if (esIndex - edIndex + 1 >= length) {
        //    sampleFrames(edIndex, esIndex, length);
        //}
        //else {
        //    padFrames(edIndex, esIndex, length);
        //}

        // 确保帧顺序正确
        ensureFrameOrder(pairFinalSelectIdx);
    }
    else {
        // 没有匹配的索引对
        int idx = keyframeIdx.back();
        if (idx >= 0) { // ED
            padFrames(idx, std::min(static_cast<int>(videoSize - 1), idx + length - 1), length);
        }
        else { // ES
            int absIdx = abs(idx);
            if (absIdx + 1 >= length) {
                padFrames(absIdx - length + 1, absIdx, length);
            }
            else {
                padFrames(0, absIdx, length);
            }
        }
    }
    //for (auto& mat : outputVideo)
    //{
    //    cv::imshow("after", mat);
    //    cv::waitKey(0);
    //}

    QtLogger::instance().logMessage(QString("[I] Parsing keyframe Success."));
    return 1; // 成功
}
