//
// Created by 单淳劼 on 2022/7/14.
//

#include "keyframeinfer.h"

KeyframeInfer::KeyframeInfer(std::string& trtEnginePath, cv::Size& inputSize, int singleVideoFrameCount, std::vector<float> Mean)
    : mEnginePath(trtEnginePath)
{
    mcvInputSize = inputSize;
    mSingleVideoFrameCount = singleVideoFrameCount;
    mInputSize = 1 * mSingleVideoFrameCount * 3 * mcvInputSize.height * mcvInputSize.width; // `1` & `3` means batch_size and channels//总像素数量
    mOutputSize = 1 * singleVideoFrameCount;

    mMean = { Mean[0], Mean[1], Mean[2] };
    initialize();
}

KeyframeInfer::~KeyframeInfer()
{
    delete mContext;
    delete mEngine;
    delete mRuntime;
}

int KeyframeInfer::initialize()
{
    cudaSetDevice(0);
    char* trtModelStream{ nullptr };
    size_t size{ 0 };

    std::ifstream file(mEnginePath, std::ios::binary);
    std::cout << "[I] Detection model creating...\n";
    if (file.good())
    {
        // 获取文件大小
        file.seekg(0, file.end);// //基地址为文件结束处，偏移地址为0，于是file.end指针定位在文件结束处
        size = file.tellg();//file为定位指针，因为它在文件结束处，所以也就是文件的大小
        file.seekg(0, file.beg);//file.beg表示输入流的开始位置
        // 分配内存存储模型文件内容size大小
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);//(起始地址，大小)
        file.close();
    }
    // 调用内部函数 createInferRuntime_INTERNAL，返回推断运行时对象的指针
    mRuntime = createInferRuntime(mGLogger);//创建一个IRuntime类，用来记录日志等
    assert(mRuntime != nullptr);

    std::cout << "[I] Key frame extraction engine creating...\n";
    mEngine = mRuntime->deserializeCudaEngine(trtModelStream, size);//读取已经序列化的文件
    assert(mEngine != nullptr);
    mContext = mEngine->createExecutionContext();//创建context来开辟空间存储中间值
    assert(mContext != nullptr);
    delete[] trtModelStream;

    auto out_dims = mEngine->getBindingDimensions(1);//bindings是tensorRT对输入输出张量的描述，bindings = input-tensor + output-tensor。比如input有a，output有b, c, d，那么bindings = [a, b, c, d]，bindings[0] = a，bindings[2] = c。此时看到engine->getBindingDimensions(0)你得知道获取的是什么


    mBlob = new float[mInputSize];
    mProb = new float[mOutputSize];

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    //指向要传递到引擎的输入和输出设备缓冲区的指针。
    //引擎正好需要 IEngine：：getNbBindings（） 缓冲区的数量。
    assert(mEngine->getNbBindings() == 2);//获取网络输入输出数量
    std::cout << "[I] Cuda buffer creating...\n";

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    //为了绑定缓冲区，我们需要知道输入和输出张量的名称。
    //请注意，索引保证小于 IEngine：：getNbBindings（）
    //    "step1.指定输入和输出节点名来获取输入输出索引"

    mInputIndex = mEngine->getBindingIndex("input");//getBindingIndex自己设定的名字
 
    assert(mEngine->getBindingDataType(mInputIndex) == nvinfer1::DataType::kFLOAT);
    mOutputIndex = mEngine->getBindingIndex("output");
    assert(mEngine->getBindingDataType(mOutputIndex) == nvinfer1::DataType::kFLOAT);
    int mBatchSize = mEngine->getMaxBatchSize();
       //"step2.在设备上开辟网络输入输出需要的GPU缓冲区(内存)"
    // Create GPU buffers on device
    checkStatus(cudaMalloc(&mBuffers[mInputIndex], mInputSize * sizeof(float)));
    checkStatus(cudaMalloc(&mBuffers[mOutputIndex], mOutputSize * sizeof(float)));//开辟显存空间，float是32位，有可能需要改变

    // Create stream
    //"step3.创建流"
    std::cout << "[I] Cuda stream creating...\n";
    checkStatus(cudaStreamCreate(&mStream));

    std::cout << "[I] Key frame extraction engine created!\n";

    return 0;
}

std::vector<int> KeyframeInfer::doInference(std::vector<cv::Mat>& sVideoPath, int sampleStride)//单个视频？
{
    std::vector<std::vector<int>> vVideosIdx;
    auto vFrames = videoInput(sVideoPath, sampleStride, vVideosIdx);
    std::vector<float> vMeans;
    // 执行推断
    doInference(vFrames, vVideosIdx, vMeans);
    // 通过均值向量vMeans找到关键帧索引
    std::vector<int> keyframes = findPeak(vMeans, vMeans.size());
    return keyframes;
}


int KeyframeInfer::doInference(std::vector<cv::Mat>& vFrames,
    std::vector<std::vector<int>>& vVideosIdx,
    std::vector<float>& vMeans)
{
    std::vector<std::vector<float>> vProbs;
    std::cout << "[I] There will be " << vVideosIdx.size() << " times inference.\n";
    for (auto& idx : vVideosIdx)
    {
        std::vector<cv::Mat> tempFrames;
        for (int index : idx)
        {
            tempFrames.push_back(vFrames[index - 1]);
        }
        //std::cout << std::endl;
        // 执行单帧推断，将推断结果存储到概率向量
        vProbs.push_back(doSingleInfer(tempFrames));
    }
    // 根据所有视频的推断概率和帧索引信息生成最终的均值向量
    vMeans = decodeOutputs(static_cast<int>(vFrames.size()), vProbs, vVideosIdx);

    for (float i : vMeans)
    {
        std::cout << i << " ";
    }

    std::cout << std::endl;
    //
    return 0;
}


std::vector<cv::Mat> KeyframeInfer::videoInput(std::vector<cv::Mat> frames, int sampleStride, std::vector<std::vector<int>>& vVideosIdx)
{
    std::vector<cv::Mat> vFrames;
    for (auto frame : frames) {

        cv::resize(frame, frame, mcvInputSize);//就变成inputsize大小就行
        frame.convertTo(frame, CV_32FC3);//变成这种格式
        vFrames.push_back(frame);
    }
    int totalCount = frames.size();
    int numPad = 0;
    // padding
    // 计算需要填充的帧数
    if (totalCount % mSingleVideoFrameCount != 0)
    {
        numPad = mSingleVideoFrameCount - (totalCount % mSingleVideoFrameCount);
    }

    if (numPad)
    {
        for (int i = 0; i < numPad; i++)
        {
            // 填充帧数据为全零图像
            vFrames.push_back(cv::Mat::zeros(mcvInputSize, CV_32FC3));
        }
    }
    // 生成单个视频的帧索引
    genVideosIdx(vFrames.size(), sampleStride, vVideosIdx);

    return vFrames;
}
//关键帧变化，
int KeyframeInfer::genVideosIdx(int totalFrameCount, int sampleStride, std::vector<std::vector<int>>& vVideosIdx) const
{
    int totalVideosCount = (totalFrameCount - mSingleVideoFrameCount + 1 + (sampleStride - 1)) / sampleStride;//我猜是滑动步长

    vVideosIdx.resize(totalVideosCount);//改变容器大小
    int counter = 1;
    int temp = 0;

    for (auto& vSingleIdx : vVideosIdx)
    {
        int tempCounter = counter;
        // 生成单个视频的帧索引
        for (int i = 0; i < mSingleVideoFrameCount; i++)
        {
            std::cout << tempCounter << " ";
            vSingleIdx.push_back(tempCounter);
            tempCounter++;
        }
        std::cout << std::endl;

        counter += sampleStride;//滑倒哪去
    }

    return 0;
}


std::vector<float> KeyframeInfer::decodeOutputs(int totalFrameCount, std::vector<std::vector<float>>& vProbs, std::vector<std::vector<int>>& vVideosIdx)
{
    std::vector<std::vector<float>> vFrameMean(totalFrameCount);
    std::vector<float> vMeanValue;

    for (int i = 0; i < totalFrameCount; i++)
    {
        int idxCounter = 0;
        //        std::cout << "No." << i << std::endl;
        for (auto& idx : vVideosIdx)
        {
            // 在当前视频的帧索引中查找当前帧的位置
            auto iter = std::find(idx.begin(), idx.end(), i + 1);
            // 如果找到，则将对应位置的推断概率添加到均值向量
            if (iter != idx.end())
            {
                auto pos = static_cast<int>(iter - idx.begin());
                vFrameMean[i].push_back(vProbs[idxCounter][pos]);
                //                std::cout << vProbs[idxCounter][pos] << " ";
            }
            //            std::cout << std::endl;
            ++idxCounter;
        }
    }
    //
    //    for (auto& mean : vFrameMean)
    //    {
    //        for (float i : mean)
    //        {
    //            std::cout << i << " ";
    //        }
    //        std::cout << std::endl;
    //    }
        // 遍历每帧的均值向量，进行单帧输出的解码，生成最终均值向量?
    for (auto& frame : vFrameMean)
    {
        vMeanValue.push_back(decodeSingleOutput(frame));//该函数的作用是对输入的概率向量进行解码，返回其元素的平均值。这种平均值的计算可能用于获得概率分布的中心趋势或其他统计信息。//纵向取均值
    }
    return vMeanValue;

}


std::vector<float> KeyframeInfer::doSingleInfer(std::vector<cv::Mat>& vSingleVideo)
{
    std::vector<float> vProb(16, 0.0f);//

    // turn std::vector<cv::Mat> to a float array格式转换
    blobFromVideo(vSingleVideo);

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    /*auto start = std::chrono::system_clock::now();*/
    //异步传输输入数据到设备，执行推断，异步传输输出数据回主机
   // "step1.拷贝数据 从主机(CPU)--->设备(GPU)"
    checkStatus(cudaMemcpyAsync(mBuffers[mInputIndex], mBlob, mInputSize * sizeof(float), cudaMemcpyHostToDevice, mStream));
   //"step2.执行推理"
    mContext->enqueueV2(mBuffers, mStream, nullptr);
    //    "step3.拷贝数据 从设备(GPU)--->主机(CPU)"

    checkStatus(cudaMemcpyAsync(mProb, mBuffers[mOutputIndex], mOutputSize * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    cudaStreamSynchronize(mStream);//数据传输
    ////因为上面的cudaMemcpyAsync()函数是异步方式执行的,所有这里需要进行同步
    //auto end = std::chrono::system_clock::now();

    for (int i = 0; i < 16; i++)
    {
        // 将推断结果从数组拷贝到概率向量
        vProb[i] = mProb[i];
        //std::cout << mProb[i] << " ";
    }
    /*std::cout << std::endl;

    std::cout << "[I] Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;*/

    return vProb;

}

void KeyframeInfer::blobFromVideo(std::vector<cv::Mat>& vSingleVideo)
{
    int counter = 0;

    // transpose Mat data to float array.
    for (auto& mat : vSingleVideo)
    {
        int row = mat.rows;
        int col = mat.cols;
        int channel = mat.channels();
        // 遍历图像通道、行和列
        for (int c = 0; c < channel; c++)
        {
            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    // 将像素值转换为浮点数，并减去均值
                    mBlob[counter * row * col * channel + c * row * col + i * col + j] = mat.at<cv::Vec3f>(i, j)[c] - mMean[c];//算法层面归一化，每个通道的值都暻秀归一化
                }
            }
        }
        ++counter;
    }
}


void KeyframeInfer::decodeSingleOutput(std::vector<float>& vProb, std::vector<int>& vIdxMax, std::vector<int>& vIdxMin)
{
    findPeaks(vProb, vIdxMax, vIdxMin);
}


void KeyframeInfer::findPeaks(std::vector<float>& vProb, std::vector<int>& vIdxMax, std::vector<int>& vIdxMin)
{
    std::vector<int> vSign;
    // 计算概率变化的符号，1 表示增加，-1 表示减少，0 表示相同
    for (int i = 1; i < vProb.size(); i++)
    {
        float diff = vProb[i] - vProb[i - 1];

        // make difference
        if (diff > 0.0f)
            vSign.push_back(1);

        else if (diff < 0.0f)
            vSign.push_back(-1);

        else
            vSign.push_back(0);
    }

    for (int i = 1; i < vSign.size(); i++)
    {
        int diff = vSign[i] - vSign[i - 1];
        //        std::cout << "i: " << i << std::endl;
        //        std::cout << "diff: " << diff << std::endl;
                // 当符号变化为负数时，表示峰值
        if (diff < 0)
        {
            vIdxMax.push_back(i);
            //            std::cout << i << std::endl;
        }

        else if (diff > 0)
            vIdxMin.push_back(i);
    }

    //    for (auto& sign : vSign)
    //        std::cout << sign << " ";

    //    std::cout << "Peaks: ";
    //    for (int i = 0; i < vIdxMax.size(); i++)
    //        std::cout << vIdxMax[i] << " ";
    //    std::cout << std::endl;
    //
    //    std::cout << "Valley: ";
    //    for (int i = 0; i < vIdxMin.size(); i++)
    //        std::cout << vIdxMin[i] << " ";
    //    std::cout << std::endl;
}

std::vector<int> KeyframeInfer::findPeak(std::vector<float> num, int count)
{
    std::vector<int> sign;
    for (int i = 1; i < count; i++)
    {
        /*相邻值做差：
         *小于0，赋-1
         *大于0，赋1
         *等于0，赋0
         */
        float diff = num[i] - num[i - 1];
        if (diff > 0)
        {
            sign.push_back(1);
        }
        else if (diff < 0)
        {
            sign.push_back(-1);
        }
        else
        {
            sign.push_back(0);
        }
    }
    //再对sign相邻位做差  
    //保存极大值和极小值的位置  
    std::vector<int> indMax;
    for (int j = 1; j < sign.size(); j++)
    {
        int diff = sign[j] - sign[j - 1];
        if (diff < 0)
        {
            indMax.push_back(j);
        }
    }
    return indMax;
}


//int main()
//{
//	std::string enginePath = "G:\\zzy2\\Drum\\application\\model\\gpu\\keyframe\\keyframe.engine";
//	cv::Size inputSize(334, 334);
//	std::string videoPath = "../b.avi";
//
//	Infer infer(enginePath, inputSize, 16);
//	auto vMeans = infer.doInference(videoPath, 1);
//	std::vector<int> keyframe = findPeak(vMeans, vMeans.size());
// 	return 0;
//}


