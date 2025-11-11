#include "view_classification_inferer.h"


ViewClassificationInferer::ViewClassificationInferer(std::string trtEnginePath, cv::Size inputSize, int singleVideoFrameCount, int inferenceTime)
    : mEnginePath(trtEnginePath)
    , mcvInputSize(inputSize)
    , mSingleVideoFrameCount(singleVideoFrameCount)
    , mInputSize(1 * mSingleVideoFrameCount * 3 * mcvInputSize.height * mcvInputSize.width) // `1` & `3` means batch_size and channels
    , mOutputSize(1 * 10)
    , mFrameNum(64)
    , mClassNum(10)
    , mInferenceTime(2)
{
    // { 114.7748, 107.7354, 99.475 }
    initialize();
}


int ViewClassificationInferer::initialize()
{
    cudaSetDevice(0);
    char *trtModelStream{ nullptr };
    size_t size{ 0 };

    std::ifstream file(mEnginePath, std::ios::binary);
    std::cout << "[I] Loading engine file from " << mEnginePath << std::endl;
    if (file.good())
    {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    mRuntime = createInferRuntime(mGLogger);
    assert(mRuntime != nullptr);

    std::cout << "[I] View classification engine creating...\n";
    mEngine = mRuntime->deserializeCudaEngine(trtModelStream, size);
    assert(mEngine != nullptr);
    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);
    delete[] trtModelStream;

    auto out_dims = mEngine->getBindingDimensions(1);

    mBlob = new float[mInputSize];
    mProb = new float[mOutputSize];

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(mEngine->getNbBindings() == 2);
    std::cout << "[I] Cuda buffer creating...\n";

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    mInputIndex = mEngine->getBindingIndex("input");

    assert(mEngine->getBindingDataType(mInputIndex) == nvinfer1::DataType::kFLOAT);
    mOutputIndex = mEngine->getBindingIndex("output");
    assert(mEngine->getBindingDataType(mOutputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = mEngine->getMaxBatchSize();

    // Create GPU buffers on device
    checkStatus(cudaMalloc(&mBuffers[mInputIndex], mInputSize * sizeof(float)));
    checkStatus(cudaMalloc(&mBuffers[mOutputIndex], mOutputSize * sizeof(float)));

    // Create stream
    std::cout << "[I] Cuda stream creating...\n";
    checkStatus(cudaStreamCreate(&mStream));

    std::cout << "[I] View classification engine created!\n";

    return 1;
}

int ViewClassificationInferer::genVideosIdx(int totalFrameCount, std::vector<int> num, std::vector<std::vector<int>>& vVideosIdx) const
{
    vVideosIdx.resize(this->mInferenceTime);
    int i = 0;
    for (auto& vSingleIdx : vVideosIdx)
    {
        int tempCounter = num[i];
        for (int i = 0; i < mSingleVideoFrameCount; i++)
        {
            vSingleIdx.push_back(tempCounter);
            tempCounter++;
        }
        i++;
    }

    return 0;
}

std::vector<cv::Mat>
ViewClassificationInferer::videoInput(std::vector<cv::Mat> video, std::vector<std::vector<int>> &vVideosIdx)
{
    std::vector<int> num;
    cv::Mat img_temp(video[0].size(), video[0].type(), cv::Scalar(0));
    std::vector<cv::Mat> video_w = video;
    int video_w_original_size = video_w.size();
    int video_w_padded_size = this->mFrameNum + this->mInferenceTime;
    if (video_w_original_size < video_w_padded_size) {
        int insert_num = video_w_padded_size - video_w_original_size;
        // 前段填充
        int insert_s = insert_num / 2;
        for (int i = 0; i < insert_s; i++) {
            video_w.insert(video_w.begin(), img_temp.clone());
        }
        // 后段填充
        for (int i = 0; i < insert_num - insert_s; i++) {
            video_w.insert(video_w.end(), img_temp.clone());
        }
    }

    // // 随机采样
    //for (int i = 0; i < this->mInferenceTime; i++) {
    //	num.push_back(random(0, video_w.size()-this->mFrameNum -1));
    //}

    // 均匀采样
    float gap = (video_w.size()-this->mFrameNum-1) / (this->mInferenceTime - 1);
    for (int i = 0; i < this->mInferenceTime; i++) {
        num.push_back(int(gap*i));
    }

    genVideosIdx(video_w.size(), num, vVideosIdx);

    return video_w;
}

void ViewClassificationInferer::blobFromVideo(std::vector<cv::Mat> &vSingleVideo)
{
    long int counter = 0;

    // transpose Mat data to float array.
    // CNN+LSTM: T,C,H,W;    SlowFast: C,T,H,W
    std::cout << "start blobbing\n";
    for (auto& mat : vSingleVideo)
    {
        cv::resize(mat, mat, mcvInputSize);
        long int rows = mat.rows;
        long int cols = mat.cols;
        long int channels = mat.channels();
        long int img_channel_pixs = rows * cols;
        long int allframes_channel_pixs = this->mFrameNum * img_channel_pixs;
        // this->mFrameNum
        for (long int c = 0; c < channels; c++)
        {
            for (long int row = 0; row < rows; row++)
            {
                for (long int col = 0; col < cols; col++)
                {
                    //mBlob[counter * img_pixs + c * channel_pixs + i * col + j] = mat.at<cv::Vec3f>(i, j)[c] - this->mMean[c];
                    // Pytorch版本SlowFast默认归一化到了-1.0~1.0
                    mBlob[c*allframes_channel_pixs + counter*img_channel_pixs + row*rows + col] = (mat.at<cv::Vec3f>(row, col)[c] - this->mMean[c])/(256.0/2.0);
                }
            }
        }
        ++counter;
    }
    std::cout << "end blobbing\n";
}

//寻找N次中重复最多的结果
int MaxFreq_index(std::vector<int> a, int n)
{
    std::map<int, int> mp;
    int i, maxfreqNum = 0;

    for (i = 0; i < n; i++)
        if (++mp[a[i]] >= mp[0])
            maxfreqNum = a[i];

    return maxfreqNum;
}

// 根据键值对中val对应的vector长度逆序排序
bool cmp_by_value(const int2ints& lhs, const int2ints& rhs)
{
    return lhs.second.size() > rhs.second.size();
};

// 所预测的帧索引后处理
view_fidxs
ViewClassificationInferer::fidxsPostprocess(view_fidxs view_frameidxs, int video_size)
{
    //view_fidxs viewresults;
    view_fidxs view_sort_frameidxs;
    view_fidxs view_w_frameidxs = view_frameidxs;
    int video_w_padded_size = this->mFrameNum + this->mInferenceTime;
    int padded_size = (this->mFrameNum + this->mInferenceTime) - video_size;
    int frameoffset = padded_size > 0 ? (padded_size / 2) : 0;
    for (int i = 0; i < view_w_frameidxs.size(); i++)
    {
        auto viewresult = view_w_frameidxs[i];
        for (int frameidx = 0; frameidx < viewresult.second.size(); frameidx++)
        {
            view_w_frameidxs[i].second[frameidx] -= frameoffset;
        }
    }

    for (int i = 0; i < view_w_frameidxs.size(); i++)
    {
        int viewidx = view_w_frameidxs[i].first;
        std::vector<int> viewframes = view_w_frameidxs[i].second;
        std::vector<int> frameidxs;
        for (int frameidx = 0; frameidx < viewframes.size(); frameidx++)
        {
            int idxstart = viewframes[frameidx];
            for (int j = idxstart; j < idxstart + this->mFrameNum; j++) {
                if ((j >= 0) && (j <= video_size - 1))
                {
                    frameidxs.push_back(j);
                }
            }
        }
        // 去重
        std::set<int> swap_temp(frameidxs.begin(), frameidxs.end());
        frameidxs.assign(swap_temp.begin(), swap_temp.end());
        // 计算连通情况（与前一索引同一连通域）
        i_i_map clipidxs;
        int connectlen = 5;
        for (int i = 0; i < frameidxs.size(); i++)
        {
            int idx = frameidxs[i];
            bool firsttime = true;
            for (int j=0; j < connectlen; j++)
            {
                if (~(clipidxs.find(idx-j) == clipidxs.end())) {
                    firsttime = false;
                    clipidxs[idx] = clipidxs[idx-j];
                    break;
                }
            }
            if (firsttime)
            {
                clipidxs[idx] = idx;
            }
        }
        // 获取最大连通集
        clip_fidxs clipfidxs;
        for (auto clipidx : clipidxs)
        {
            if ((clipfidxs.find(clipidx.second) == clipfidxs.end())) {
                std::vector<int> temp;
                clipfidxs[clipidx.second] = temp;
            }
            clipfidxs[clipidx.second].push_back(clipidx.first);
        }
        view_fidxs tempfidxs(clipfidxs.begin(), clipfidxs.end());
        sort(tempfidxs.begin(), tempfidxs.end(), cmp_by_value);
        view_sort_frameidxs.push_back(std::make_pair(viewidx, tempfidxs[0].second));
    }



    return view_sort_frameidxs;
    //return viewresults;
}


std::vector<float> ViewClassificationInferer::doSingleInfer(std::vector<cv::Mat> &vSingleVideo)
{
    std::vector<float> vProb(this->mClassNum, 0.0f);

    // turn std::vector<cv::Mat> to a float array
    blobFromVideo(vSingleVideo);

    //cv::imshow("a", vSingleVideo[0]);
    //cv::waitKey();

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    // auto start = std::chrono::system_clock::now();
    checkStatus(cudaMemcpyAsync(mBuffers[mInputIndex], mBlob, mInputSize * sizeof(float), cudaMemcpyHostToDevice, mStream));
    auto start = std::chrono::system_clock::now();
    mContext->enqueueV2(mBuffers, mStream, nullptr);
    checkStatus(cudaMemcpyAsync(mProb, mBuffers[mOutputIndex], mOutputSize * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    cudaStreamSynchronize(mStream);
    auto end = std::chrono::system_clock::now();

    for (int i = 0; i < this->mClassNum; i++)
    {
        vProb[i] = mProb[i];
        //std::cout << mProb[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "[I] Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

    return vProb;

}

view_fidxs ViewClassificationInferer::Inference(std::vector<cv::Mat> &vFrames,
                                std::vector<std::vector<int>> &vVideosIdx)
{
    std::cout << "ViewClassificationInferer::Inference\n";
    int doInferenceTime = this->mInferenceTime;
    float output_th = 1.0 / 4.0;
    std::vector<std::vector<float>> vProbs;
    std::cout << "[I] There will be " << vVideosIdx.size() << " times inference.\n";
    for (auto& idx : vVideosIdx)
    {
        std::vector<cv::Mat> tempFrames;
        for (int index : idx)
        {
            //cv::Mat temp = vFrames[index - 1].clone();
            cv::Mat temp = vFrames[index].clone();
            temp.convertTo(temp, CV_32FC3);
            tempFrames.push_back(temp);
        }
        std::cout << std::endl;
        std::cout << "start single infer.\n";

        vProbs.push_back(doSingleInfer(tempFrames));
    }
    std::vector<int> final;
    for (int i = 0; i < doInferenceTime; i++) {
        final.push_back(max_element(vProbs[i].begin(), vProbs[i].end()) - vProbs[i].begin());
    }

    // 统计各视频段预测结果
    clip_fidxs clipresult;
    for (int i = 0; i < final.size(); i++)
    {
        int classIdx = final[i];
        if (clipresult.find(classIdx) == clipresult.end()) {
            std::vector<int> temp;
            clipresult[classIdx] = temp;
        }
        clipresult[classIdx].push_back(vVideosIdx[i][0]);
    }

    // 根据各预测切面的视频段数目逆序排序，得到预测结果（视频段数目从高到低）
    view_fidxs tempresult(clipresult.begin(), clipresult.end());
    sort(tempresult.begin(), tempresult.end(), cmp_by_value);

    // 默认输出所有高于1/3总段数的预测类别，或者输出最多视频段对应的切面类型
    view_fidxs finalresult;
    for (int i = 0; i < tempresult.size(); i++)
    {
        if (tempresult[i].second.size() >= floor(float(doInferenceTime)*output_th))
        {
            finalresult.push_back(tempresult[i]);
        }
    }
    if (finalresult.size() < 1)
    {
        finalresult.push_back(tempresult[0]);
    }

    //int finalresult = MaxFreq_index(final, doInferenceTime);
    return finalresult;
}

view_fidxs
ViewClassificationInferer::doInference(std::vector<cv::Mat> video)
{
    std::vector<std::vector<int>> vVideosIdx;
    auto vFrames = videoInput(video, vVideosIdx);
    std::cout << "Video input." << std::endl;

    view_fidxs view_w_frameidxs = Inference(vFrames, vVideosIdx);
    std::cout << "[I] Inference phase ends.\n";

    view_fidxs viewresults = fidxsPostprocess(view_w_frameidxs, video.size());

    return viewresults;
}
