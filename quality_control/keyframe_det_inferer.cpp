#include "keyframe_det_inferer.h"

KeyframeDetInferer::KeyframeDetInferer(std::string sBackboneEnginePath, std::string sSGTAEnginePath, int staLength, int memoryLength, bool isEDonly, bool isAvgProb,
    std::vector<float> vecMeans, std::vector<float> vecStds)
    : m_inputDims({1, 3, 320, 320})
    , m_featDims({1, 512, 10, 10})
    , m_inputMemDims({memoryLength, 512, 10, 10})
    , m_inputQueryDims({staLength, 512, 10, 10})
    , m_bIsEDOnly(isEDonly)
    , m_bIsAvgProb(isAvgProb)
    //, m_outputProbDims({1, 2})
    , m_staLength(staLength)
    , m_memLength(memoryLength)
    , m_means(vecMeans)
    , m_stds(vecStds)
{
    if (m_bIsEDOnly && m_bIsAvgProb)
        m_outputProbDims = { 1, 5 };
    else if (m_bIsEDOnly && !m_bIsAvgProb)
        m_outputProbDims = { 1, 1 };
    else if (!m_bIsEDOnly && m_bIsAvgProb)
        m_outputProbDims = { 1, 10 };
    else
        m_outputProbDims = { 1, 2 };

    initializeBackboneEngine(sBackboneEnginePath);
    initializeSGTAEngine(sSGTAEnginePath);
}

int KeyframeDetInferer::initializeBackboneEngine(std::string sBackboneEnginePath)
{
    std::ifstream engineFile(sBackboneEnginePath, std::ios::binary);
    if (engineFile.fail())
    {
        return 0;
    }

    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger::gLogger.getTRTLogger());
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
    assert(engine != nullptr);

    m_backboneContext = engine->createExecutionContext();
    assert(m_backboneContext != nullptr);

    m_inputImageIdx = engine->getBindingIndex("image");
    m_outputFeatIdx = engine->getBindingIndex("feat");

    //m_backboneContext->setBindingDimensions(m_inputImageIdx, m_inputDims);
}

int KeyframeDetInferer::initializeSGTAEngine(std::string sSGTAEnginePath)
{
    std::ifstream engineFile(sSGTAEnginePath, std::ios::binary);
    if (engineFile.fail())
    {
        return 0;
    }

    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger::gLogger.getTRTLogger());
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
    assert(engine != nullptr);

    m_sgtaContext = engine->createExecutionContext();
    assert(m_sgtaContext != nullptr);

    m_inputMemFeatIdx = engine->getBindingIndex("memory_feat");
    m_inputQueryFeatIdx = engine->getBindingIndex("query_feat");
    m_outputProbIdx = engine->getBindingIndex("prob");

    //m_sgtaContext->setBindingDimensions(m_inputMemFeatIdx, m_inputMemDims);
    //m_sgtaContext->setBindingDimensions(m_inputQueryFeatIdx, m_inputQueryDims);
}

int KeyframeDetInferer::doInference(cv::Mat& src, std::string mode, std::vector<float>& vProbs)
{
    // std::vector<float> vProbs;
    if (mode == "backbone")
    {
        float* blob = new float[getElementNum(m_inputDims)];
        float* feat = new float[getElementNum(m_featDims)];
        floatArrayPtr currFeatPtr = floatArrayPtr(feat);
        preprocess(src, blob);
        doBackboneInference(blob, currFeatPtr.get());

        m_vFeats.push_back(floatArrayPtr(currFeatPtr));
        delete[] blob;
    }

    else if (mode == "sgta")
    {
        std::vector<float> vecESProbs;
        sgtaInference(src, vProbs, vecESProbs);
    }

    return 1;
}

int KeyframeDetInferer::doInference(cv::Mat& src, std::string mode, std::vector<float>& vecEDProbs, std::vector<float>& vecESProbs)
{
    //std::lock_guard<std::mutex> lock(m_featsMutex);
    if (mode == "backbone")
    {
        float* blob = new float[getElementNum(m_inputDims)];
        float* feat = new float[getElementNum(m_featDims)];
        floatArrayPtr currFeatPtr = floatArrayPtr(feat);
        preprocess(src, blob);
        doBackboneInference(blob, currFeatPtr.get());

        m_vFeats.push_back(floatArrayPtr(currFeatPtr));
        delete[] blob;
    }

    else if (mode == "sgta")
    {
        sgtaInference(src, vecEDProbs, vecESProbs);
    }
    return 1;
}

int KeyframeDetInferer::doInference(cv::Mat& src, std::string mode, std::vector<PeakInfo>& vPeaks)
{
    std::vector<float> vProbs, vecEDProbs, vecESProbs;
    std::vector<PeakInfo> tempEdIndex;

    m_demoImages.push_back(src);
    if (mode == "backbone")
    {
        doInference(src, mode, vProbs);
    }
    else
    {
        if (m_bIsEDOnly)
        {
            doInference(src, mode, vProbs);
            tempEdIndex = findPeaks(vProbs, 0.5f, 8, 0.01f, 0);
            for (auto& v : vProbs)
            {
                v = -v;
            }
            vPeaks = findPeaks(vProbs, -0.5f, 8, 0.01f, 0);
            for (auto& v : vPeaks)
            {
                v.index = -v.index;
            }
            for (auto& v : tempEdIndex)
            {
                vPeaks.push_back(v);
            }
        }
            
        else
        {
            doInference(src, mode, vecEDProbs, vecESProbs);
            tempEdIndex = findPeaks(vecEDProbs, 0.5f, 8, 0.01f, 0);
            vPeaks = findPeaks(vecESProbs, 0.5f, 8, 0.01f, 0);
            for (auto& v : vPeaks)
            {
                v.index = -v.index;
            }
            for (auto& v : tempEdIndex)
            {
                vPeaks.push_back(v);
            }
        }
    }

    return 1;
}

int KeyframeDetInferer::doBackboneInference(float* blob, float* feat)
{
    void* inputMem{ nullptr };
    void* outputMem{ nullptr };
    size_t inputSize = getMemorySize(m_inputDims, sizeof(float));
    size_t outputSize = getMemorySize(m_featDims, sizeof(float));

    if (cudaMalloc(&inputMem, inputSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: input cuda memory allocation failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputMem, outputSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(inputMem, blob, inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    // Run TensorRT inference
    void* bindings[] = { inputMem, outputMem };
    bool status = m_backboneContext->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(feat, outputMem, outputSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputSize << " bytes" << std::endl;
        return 0;
    }

    cudaFree(inputMem);
    cudaFree(outputMem);
    cudaStreamDestroy(stream);

    return 1;
}

int KeyframeDetInferer::doSGTAInference(float* memFeats, float* feat, float* prob)
{
    void* inputFeatsMem{ nullptr };
    void* inputQueryMem{ nullptr };
    void* outputProbMem{ nullptr };

    size_t inputFeatsSize = getMemorySize(m_inputMemDims, sizeof(float));
    size_t inputQuerySize = getMemorySize(m_inputQueryDims, sizeof(float));
    size_t outputProbSize = getMemorySize(m_outputProbDims, sizeof(float));

    if (cudaMalloc(&inputFeatsMem, inputFeatsSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: input cuda memory allocation failed, size = " << inputFeatsSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&inputQueryMem, inputQuerySize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: input cuda memory allocation failed, size = " << inputQuerySize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputProbMem, outputProbSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputProbSize << " bytes" << std::endl;
        return 0;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(inputFeatsMem, memFeats, inputFeatsSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputFeatsSize << " bytes" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(inputQueryMem, feat, inputQuerySize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputQuerySize << " bytes" << std::endl;
        return 0;
    }

    // Run TensorRT inference
    void* bindings[] = { inputFeatsMem, inputQueryMem, outputProbMem };
    bool status = m_sgtaContext->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(prob, outputProbMem, outputProbSize, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputProbSize << " bytes" << std::endl;
        return 0;
    }

    cudaFree(inputFeatsMem);
    cudaFree(inputQueryMem);
    cudaFree(outputProbMem);
    cudaStreamDestroy(stream);
    return 1;
}

int KeyframeDetInferer::calcEachFrameAvgProb(std::vector<std::vector<float>>& vecTotalProbs, std::vector<float>& vecAvgProbs)
{
    std::vector<std::vector<float>> vecEachFrameProbs(vecTotalProbs.size());
    for (int nFrameIdx = 0; nFrameIdx < vecTotalProbs.size(); ++nFrameIdx)
    {
        //for (int nQueryFrameIdx = 0; nQueryFrameIdx < m_staLength; ++nQueryFrameIdx)
        //{
        //    vecEachFrameProbs.at(nFrameIdx - nQueryFrameIdx).push_back(vecTotalProbs.at(nFrameIdx).at(nQueryFrameIdx));
        //}
        int nCounter = 0;
        for (int nQueryFrameIdx = -m_staLength / 2; nQueryFrameIdx <= m_staLength / 2; ++nQueryFrameIdx)
        {
            int nCurrSampleQueryFrameIdx = std::min(std::max(nFrameIdx + nQueryFrameIdx, 0), static_cast<int>(vecTotalProbs.size() - 1));
            vecEachFrameProbs.at(nCurrSampleQueryFrameIdx).push_back(vecTotalProbs.at(nFrameIdx).at(nCounter));
            ++nCounter;
        }
    }

    for (auto& vecSingleFrameProbs : vecEachFrameProbs)
        vecAvgProbs.push_back(calcAverageValue(vecSingleFrameProbs));

    return 1;
}

int KeyframeDetInferer::sgtaInference(cv::Mat& src, std::vector<float>& vecEDProbs, std::vector<float>& vecESProbs)
{
    float* blob = new float[getElementNum(m_inputDims)];
    float* feat = new float[getElementNum(m_featDims)];
    float* memFeats = new float[getElementNum(m_inputMemDims)];

    preprocess(src, blob);
    doBackboneInference(blob, feat);

    std::lock_guard<std::mutex> lock(m_featsMutex);
    m_vFeats.push_back(floatArrayPtr(feat));

    int ret = getSampledElements(m_memLength);
    if (!ret)
    {
        delete[] memFeats;
        delete[] blob;
        return 0;
    }

    blobFromFeats(memFeats);

    std::vector<std::vector<float>> vecTotalEDProbs, vecTotalESProbs;

    for (int i = 0; i < m_vFeats.size(); i++)
    {
        float* queryFeat = new float[getElementNum(m_featDims) * m_staLength];
        float* prob = new float[getElementNum(m_outputProbDims)];

        std::vector<float*> sampledFeats = getQueryFeat(i);
        
        if (sampledFeats.empty())
        {
            delete[] queryFeat;
            delete[] prob;
            break;
        }

        blobFromQueryFeats(sampledFeats, queryFeat);
        doSGTAInference(memFeats, queryFeat, prob);

        if (!m_bIsAvgProb)
        {
            if (m_bIsEDOnly)
                vecEDProbs.push_back(*prob);
            else
            {
                vecEDProbs.push_back(prob[1]);
                vecESProbs.push_back(prob[0]);
            }
        }

        else
        {
            if (m_bIsEDOnly)
            {
                std::vector<float> vecCurrProb;
                for (int j = 0; j < m_staLength; ++j)
                    vecCurrProb.push_back(prob[j]);
                vecTotalEDProbs.push_back(vecCurrProb);
            }
            else
            {
                std::vector<float> vecCurrEDProb, vecCurrESProb;
                for (int j = 0; j < m_staLength; ++j)
                {
                    vecCurrEDProb.push_back(prob[j * 2]);
                    vecCurrESProb.push_back(prob[j * 2 + 1]);
                }
                vecTotalEDProbs.push_back(vecCurrEDProb);
                vecTotalESProbs.push_back(vecCurrESProb);
            }
        }

        delete[] queryFeat;
        delete[] prob;
    }

    if (m_bIsAvgProb)
    {
        if (m_bIsEDOnly)
            calcEachFrameAvgProb(vecTotalEDProbs, vecEDProbs);
        else
        {
            calcEachFrameAvgProb(vecTotalEDProbs, vecEDProbs);
            calcEachFrameAvgProb(vecTotalESProbs, vecESProbs);
        }
    }

    delete[] memFeats;
    delete[] blob;
    releaseFeatsVector();
    return 1;
}

float KeyframeDetInferer::checkEDESClose(std::vector<PeakInfo>& vecEDPeaks, std::vector<PeakInfo>& vecESPeaks)
{
    if (vecEDPeaks.empty() || vecESPeaks.empty())
        return 0.0f;

    float fMinDiff = 10000.0f;
    for (auto& peakED : vecEDPeaks)
    {
        float fEDProb = peakED.value;
        for (auto& peakES : vecESPeaks)
        {
            float fESProb = peakES.value;
            float fCurrDiff = std::abs(fEDProb - fESProb);
            if (fCurrDiff < fMinDiff)
                fMinDiff = fCurrDiff;
        }
    }

    return fMinDiff;
}

float KeyframeDetInferer::checkProbClose(std::vector<float>& vecProbs)
{
    float fMaxProb = *std::max_element(vecProbs.begin(), vecProbs.end());
    float fMinProb = *std::min_element(vecProbs.begin(), vecProbs.end());

    return fMaxProb - fMinProb;
}

int KeyframeDetInferer::preprocess(cv::Mat& src, float* blob)
{
    cv::Mat dst;

    src.convertTo(dst, CV_32FC3);
    cv::resize(dst, dst, cv::Size(m_inputDims.d[2], m_inputDims.d[3]));
    blobFromImage(dst, blob);
    return 1;
}

int KeyframeDetInferer::blobFromImage(cv::Mat& src, float* blob)
{
    int channels = src.channels();
    int rows = src.rows;
    int cols = src.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                blob[c * rows * cols + i * cols + j] = ((src.at<cv::Vec3f>(i, j)[c] / 255.0f) - m_means[c]) / m_stds[c];
            }
        }
    }

    return 1;
}

int KeyframeDetInferer::blobFromFeats(float* blobFeat)
{
    int counter = 0;
    int featElemNum = getElementNum(m_featDims);
    for (auto& feat : m_memFeats)
    {
        for (int i = 0; i < featElemNum; i++)
        {
            blobFeat[counter * featElemNum + i] = feat[i];
        }
        counter++;
    }
    return 1;
}

int KeyframeDetInferer::blobFromQueryFeats(std::vector<float*>& vSampledFeats, float* blobFeat)
{
    int counter = 0;
    int featElemNum = getElementNum(m_featDims);
    for (auto& feat : vSampledFeats)
    {
        for (int i = 0; i < featElemNum; i++)
        {
            blobFeat[counter * featElemNum + i] = feat[i];
        }
        counter++;
    }
    return 1;
}

int KeyframeDetInferer::findLocalMaxima(const std::vector<float>& vecInputs,
    std::vector<int>& vecPeakIdxes, 
    std::vector<int>& vecLeftEdges,
    std::vector<int>& vecRightEdges)
{
    if (vecInputs.empty())
        return 0;
    size_t n = vecInputs.size();

    // Preallocate, there can't be more maxima than half the size of x
    vecPeakIdxes.reserve(n / 2);
    vecLeftEdges.reserve(n / 2);
    vecRightEdges.reserve(n / 2);

    for (size_t i = 1; i < n - 1; ++i) {
        // Check if current sample is greater than the previous sample
        if (vecInputs[i - 1] < vecInputs[i]) {
            size_t i_ahead = i + 1;

            // Find the next sample that is unequal to x[i]
            while (i_ahead < n - 1 && vecInputs[i_ahead] == vecInputs[i]) {
                i_ahead++;
            }

            // Check if we found a local maximum
            if (i_ahead < n && vecInputs[i_ahead] < vecInputs[i]) {
                vecLeftEdges.push_back(i);
                vecRightEdges.push_back(i_ahead - 1);
                vecPeakIdxes.push_back((vecLeftEdges.back() + vecRightEdges.back()) / 2);

                // Skip samples that can't be maxima
                i = i_ahead;
            }
        }
    }

    if (vecPeakIdxes.empty())
    {
        if (n > 1 && vecInputs[0] > vecInputs[1]) {
            vecLeftEdges.push_back(0);
            vecRightEdges.push_back(0);
            vecPeakIdxes.push_back(0);
        }

        if (n > 1 && vecInputs[n - 1] > vecInputs[n - 2]) {
            vecLeftEdges.push_back(n - 1);
            vecRightEdges.push_back(n - 1);
            vecPeakIdxes.push_back(n - 1);
        }
    }

    return 1;
}

std::vector<PeakInfo> KeyframeDetInferer::filterPeaksByDistance(std::vector<PeakInfo>& vecPeakInfos, int distance)
{
    std::vector<PeakInfo> selectedPeaks;
    selectedPeaks.push_back(vecPeakInfos[0]);
    for (auto peak : vecPeakInfos)
    {
        if (abs(peak.index - selectedPeaks.back().index) >= distance)
        {
            selectedPeaks.push_back(peak);
        }
        else
        {
            if (selectedPeaks.back().value < peak.value)
            {
                selectedPeaks.back() = peak;
            }
        }
    }
    return selectedPeaks;
}

std::vector<PeakInfo> KeyframeDetInferer::filterPeaksByValue(std::vector<PeakInfo>& vecPeakInfos, float fHeight, float fThreshold)
{
    std::vector<PeakInfo> selectedPeaks;

    for (auto& peak : vecPeakInfos)
    {
        if (peak.value >= fHeight)
        {
            selectedPeaks.push_back(peak);
        }
    }
    return selectedPeaks;
}

std::vector<PeakInfo> KeyframeDetInferer::findPeaks(
    const std::vector<float>& input,
    float height,
    size_t distance,
    float threshold,
    size_t plateau_size
)
{
    std::vector<PeakInfo> peaks, selectedPeaks;
    if (input.size() < plateau_size + 2) // Need at least plateau_size + 2 points to find peaks
        return peaks;

    std::vector<int> vecPeakIdxes, vecLeftEdges, vecRightEdges;
    findLocalMaxima(input, vecPeakIdxes, vecLeftEdges, vecRightEdges);

    for (int idx : vecPeakIdxes)
        peaks.push_back({ idx, input[idx], 1.0f, 0.0f });

    //for (size_t i = plateau_size; i < input.size() - plateau_size; ++i) {
    //    if (input[i] > height) {
    //        if (isPeak(input, i, threshold)) {
    //            // Handle plateau
    //            size_t plateau_end = i;
    //            if (plateau_size < 1)
    //                peaks.push_back({ static_cast<int>(i), input[i], 1.0, 0.0 });
    //            while (plateau_end + 1 < input.size() && input[plateau_end] == input[plateau_end + 1]) {
    //                ++plateau_end;
    //            }
    //            // Check if plateau is not too large
    //            if (plateau_end - i + 1 <= plateau_size) {
    //                // Check minimum distance from previous peaks
    //                bool isFarEnough = std::all_of(peaks.begin(), peaks.end(),
    //                                               [i, distance](const PeakInfo& peak) { return std::abs(peak.index - static_cast<int>(i)) >= static_cast<long long>(distance); });

    //                if (isFarEnough) {
    //                    float width = plateau_end - i + 1;
    //                    // Calculate prominence if required here
    //                    float prominence = 0; // Simplified, actual calculation is more involved
    //                    peaks.push_back({static_cast<int>(i), input[i], width, prominence});
    //                }
    //            }
    //            i = plateau_end; // Skip plateau
    //        }
    //    }
    //}

    if (peaks.empty())
        return peaks;

    selectedPeaks = filterPeaksByDistance(peaks, distance);
    selectedPeaks = filterPeaksByValue(selectedPeaks, height, threshold);

    return selectedPeaks;
}

std::vector<int> KeyframeDetInferer::findPeaks(std::vector<double> x,
                                               std::vector<double> plateauSize,
                                               std::vector<double> height,
                                               std::vector<double> threshold,
                                               int distance, std::vector<double> prominence,
                                               int wlen, std::vector<double> width,
                                               double relHeight)
    // https://github.com/johnhillross/find_peaks-CPP/tree/main
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

                    if (std::min(x[currentPeakIndex] - x[currentPeakIndex - 1], x[currentPeakIndex] - x[currentPeakIndex + 1]) < threshold[0]
                        || std::max(x[currentPeakIndex] - x[currentPeakIndex - 1], x[currentPeakIndex] - x[currentPeakIndex + 1]) > threshold[1]) {
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

KeyframeLGTAInferer::KeyframeLGTAInferer(std::string strbackbonePath, std::string strLGTAEnginePath, int shortLength, int memoryLength, bool isEDOnly, bool isAvgProb,
    std::vector<float> vecMeans, std::vector<float> vecStds)
    : KeyframeDetInferer(strbackbonePath, strLGTAEnginePath, shortLength, memoryLength, isEDOnly, isAvgProb, vecMeans, vecStds)

{
    m_featDims = { 1, 1024, 10, 10 };
    m_inputMemDims = { memoryLength, 1024, 10, 10 };
    m_inputQueryDims = { shortLength, 1024, 10, 10 };

}

int KeyframeLGTAInferer::getSampledElements(int nElementCount, int nStride)
{
    if (m_vFeats.empty())
        return 0;

    if (m_memFeats.size() == m_inputMemDims.d[0])
        return 1;

    int nFeatArraySize = getElementNum(m_featDims);
    for (int i = 0; i < nElementCount * nStride; i += nStride)
    {
        //if (i > m_vFeats.size() - 1 || m_memFeats.size() == m_inputMemDims.d[0])
        //    break;
        if (m_memFeats.size() == m_inputMemDims.d[0])
            break;

        if (i > m_vFeats.size() - 1)
        {
            floatArrayPtr lastElem = m_vFeats.back();
            floatArrayPtr newElem(new float[nFeatArraySize]);

            for (int j = 0; j < nFeatArraySize; ++j)
            {
                newElem.get()[j] = lastElem.get()[j];
            }
            m_memFeats.push_back(newElem);
        }
        else
            m_memFeats.push_back(m_vFeats[i]);

    }

    return 1;
}

int KeyframeLGTAInferer::doInference(cv::Mat& src, std::string mode, std::vector<PeakInfo>& vPeaks)
{
    std::vector<float> vProbs, vecEDProbs, vecESProbs;
    std::vector<PeakInfo> tempEdIndex;

    //if (mode == "backbone")
    //{
    //    m_demoImages.push_back(src);
    //}
    //else
    //{
    //    std::vector<float> vecOutputProb;
    //    end2endInference(m_demoImages, vecOutputProb);
    //}

    m_demoImages.push_back(src);
    if (mode == "backbone")
    {
        backboneInference(src);
    }
    else
    {
        if (m_bIsEDOnly)
        {
            KeyframeDetInferer::doInference(src, mode, vProbs);
            tempEdIndex = findPeaks(vProbs, 0.5f, 8, 0.01f, 0);
            for (auto& v : vProbs)
            {
                v = -v;
            }
            vPeaks = findPeaks(vProbs, -0.5f, 8, 0.01f, 0);
            for (auto& v : vPeaks)
            {
                v.index = -v.index;
            }
            for (auto& v : tempEdIndex)
            {
                vPeaks.push_back(v);
            }
        }

        else
        {
            //doInference(src, mode, vecEDProbs, vecESProbs);
            lgtaInference(src, vecEDProbs, vecESProbs);

            float fEDDiff = checkProbClose(vecEDProbs);
            float fESDiff = checkProbClose(vecESProbs);

            if (fEDDiff < m_fEDESProbDiffThresh || fESDiff < m_fEDESProbDiffThresh)
            {
                QtLogger::instance().logMessage(QString("[E] Close Keyframe Probabilities, ED: %1, ES: %2, Skip...").arg(fEDDiff).arg(fESDiff));
                vPeaks.clear();
                return 0;
            }

            tempEdIndex = findPeaks(vecEDProbs, 0.3f, 8, 0.01f, 0);
            vPeaks = findPeaks(vecESProbs, 0.3f, 8, 0.01f, 0);


            for (auto& v : vPeaks)
            {
                v.index = -v.index;
            }
            for (auto& v : tempEdIndex)
            {
                vPeaks.push_back(v);
            }
        }
    }

    //if (!vPeaks.empty())
    //{
    //    for (int i = 0; i < m_demoImages.size(); ++i)
    //    {
    //        cv::Mat currImage = m_demoImages[i];
    //        bool bFindFlag = false;
    //        int nIndex = 0;
    //        for (auto& peak : vPeaks)
    //        {
    //            if (i == std::abs(peak.index))
    //            {
    //                bFindFlag = true;
    //                nIndex = peak.index;
    //                break;
    //            }
    //        }

    //        if (bFindFlag)
    //        {
    //            std::string strText;
    //            if (nIndex >= 0)
    //                strText = "ED";
    //            else
    //                strText = "ES";

    //            cv::putText(currImage, strText, cv::Point(20, 20), cv::FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    //        }

    //        cv::imshow("keyframe", currImage);
    //        cv::waitKey(0);
    //    }
    //}


    return 1;
}

int KeyframeLGTAInferer::doInference(cv::Mat& src, std::string mode, std::vector<float>& vecEDProbs, std::vector<float>& vecESProbs)
{
    if (mode == "backbone")
    {
        float* blob = new float[getElementNum(m_inputDims)];
        float* feat = new float[getElementNum(m_featDims)];
        floatArrayPtr currFeatPtr = floatArrayPtr(feat);
        preprocess(src, blob);
        doBackboneInference(blob, currFeatPtr.get());

        std::lock_guard<std::mutex> lock(m_featsMutex);
        m_vFeats.push_back(floatArrayPtr(currFeatPtr));
        delete[] blob;
    }

    else if (mode == "sgta")
    {
        lgtaInference(src, vecEDProbs, vecESProbs);
    }

    return 1;
}

int KeyframeLGTAInferer::lgtaInference(cv::Mat& src, std::vector<float>& vecEDProbs, std::vector<float>& vecESProbs)
{
    float* blob = new float[getElementNum(m_inputDims)];
    float* feat = new float[getElementNum(m_featDims)];
    float* memFeats = new float[getElementNum(m_inputMemDims)];

    backboneInference(src);
    //preprocess(src, blob);
    //doBackboneInference(blob, feat);

    //std::lock_guard<std::mutex> lock(m_featsMutex);
    //m_vFeats.push_back(floatArrayPtr(feat));

    int ret = getSampledElements(m_memLength, 2);
    if (!ret)
    {
        delete[] memFeats;
        delete[] blob;
        return 0;
    }

    blobFromFeats(memFeats);

    std::vector<std::vector<float>> vecTotalEDProbs, vecTotalESProbs;

    for (int i = 0; i < m_vFeats.size(); i++)
    {
        float* queryFeat = new float[getElementNum(m_inputQueryDims)];
        float* prob = new float[getElementNum(m_outputProbDims)];

        std::vector<float*> sampledFeats = getQueryFeat(i);

        if (sampledFeats.empty())
        {
            delete[] queryFeat;
            delete[] prob;
            break;
        }

        blobFromQueryFeats(sampledFeats, queryFeat);
        doSGTAInference(memFeats, queryFeat, prob);

        if (!m_bIsAvgProb)
        {
            if (m_bIsEDOnly)
                vecEDProbs.push_back(*prob);
            else
            {
                vecEDProbs.push_back(prob[1]);
                vecESProbs.push_back(prob[0]);
            }
        }

        else
        {
            if (m_bIsEDOnly)
            {
                std::vector<float> vecCurrProb;
                for (int j = 0; j < m_staLength; ++j)
                    vecCurrProb.push_back(prob[j]);
                vecTotalEDProbs.push_back(vecCurrProb);
            }
            else
            {
                std::vector<float> vecCurrEDProb, vecCurrESProb;
                int nProbMaxIdx = getElementNum(m_outputProbDims) - 1;
                for (int j = 0; j < m_staLength; ++j)
                //for (int j = 0; j < 10; ++j)
                {
                    //vecCurrEDProb.push_back(prob[j]);

                    if (j + m_staLength > nProbMaxIdx)
                        break;

                    vecCurrEDProb.push_back(prob[j]);
                    vecCurrESProb.push_back(prob[j + m_staLength]);
                }
                vecTotalEDProbs.push_back(vecCurrEDProb);
                vecTotalESProbs.push_back(vecCurrESProb);
            }
        }

        delete[] queryFeat;
        delete[] prob;
    }

    if (m_bIsAvgProb)
    {
        if (m_bIsEDOnly)
            calcEachFrameAvgProb(vecTotalEDProbs, vecEDProbs);
        else
        {
            calcEachFrameAvgProb(vecTotalEDProbs, vecEDProbs);
            calcEachFrameAvgProb(vecTotalESProbs, vecESProbs);
        }
    }

    delete[] memFeats;
    delete[] blob;
    //releaseFeatsVector();
    return 1;
}

std::vector<float*> KeyframeLGTAInferer::getQueryFeat(int index)
{
    std::vector<float*> sampledFeats;

    if (m_vFeats.empty())
        return sampledFeats;

    for (int i = m_staLength - 1; i >= 0; --i)
    {
        int currIdx = index - i;
        currIdx = (std::min)((std::max)(currIdx, 0), static_cast<int>(m_vFeats.size() - 1));
        sampledFeats.push_back(m_vFeats[currIdx].get());
    }

    return sampledFeats;
}

int KeyframeLGTAInferer::calcEachFrameAvgProb(std::vector<std::vector<float>>& vecTotalProbs, std::vector<float>& vecAvgProbs)
{
    std::vector<std::vector<float>> vecEachFrameProbs(vecTotalProbs.size());
    for (int nFrameIdx = 0; nFrameIdx < vecTotalProbs.size(); ++nFrameIdx)
    {
        int nCounter = 0;
        for (int nQueryFrameIdx = -(m_staLength - 1); nQueryFrameIdx <= 0; ++nQueryFrameIdx)
        {
            int nCurrSampleQueryFrameIdx = std::min(std::max(nFrameIdx + nQueryFrameIdx, 0), static_cast<int>(vecTotalProbs.size() - 1));
            vecEachFrameProbs.at(nCurrSampleQueryFrameIdx).push_back(vecTotalProbs.at(nFrameIdx).at(nCounter));
            ++nCounter;
        }
    }

    for (auto& vecSingleFrameProbs : vecEachFrameProbs)
        vecAvgProbs.push_back(calcAverageValue(vecSingleFrameProbs));

    return 1;
}

int KeyframeLGTAInferer::backboneInference(cv::Mat& src)
{
    float* blob = new float[getElementNum(m_inputDims)];
    float* feat = new float[getElementNum(m_featDims)];
    floatArrayPtr currFeatPtr = floatArrayPtr(feat);
    preprocess(src, blob);
    doBackboneInference(blob, currFeatPtr.get());

    std::lock_guard<std::mutex> lock(m_featsMutex);
    m_vFeats.push_back(floatArrayPtr(currFeatPtr));
    delete[] blob;

    return 1;
}
