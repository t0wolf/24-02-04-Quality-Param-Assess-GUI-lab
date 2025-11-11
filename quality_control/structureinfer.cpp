//
// Created by 单淳劼 && 曾子炀 && 成汉林 on 2022/7/18.
//

#include "structureinfer.h"


StructureInfer::StructureInfer(std::string& trtEnginePath, cv::Size& inputSize, std::vector<float> Mean, int objectNum)
    : mEnginePath(trtEnginePath)
{
    mcvInputSize = inputSize;
    mInputSize = 1 * 3 * mcvInputSize.height * mcvInputSize.width; // `1` & `3` means batch_size and channels
    mobjectNum = objectNum;
    mLabelsSize = 8732 * objectNum;
    mScoresSize = 8732 * objectNum;
    mBoxesSize = 8732 * objectNum * 4;

    vMeans = { Mean[0], Mean[1], Mean[2] };

    initialize();
}

StructureInfer::~StructureInfer()
{
    delete this->mContext;
    delete this->mRuntime;
    delete this->mEngine;

    cudaStreamDestroy(mStream);

    checkStatus(cudaFree(mBuffers[mInputIndex]));
    checkStatus(cudaFree(mBuffers[mBoxesIndex]));
    checkStatus(cudaFree(mBuffers[mScoresIndex]));
    checkStatus(cudaFree(mBuffers[mLabelsIndex]));

    delete[] this->mBlob;
    delete[] this->mScores;
    delete[] this->mBoxes;
    delete[] this->mLabels;
}

int StructureInfer::initialize()
{
    cudaSetDevice(0);
    char* trtModelStream{ nullptr };
    size_t size{ 0 };
    // initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
    std::ifstream file(mEnginePath, std::ios::binary);
    std::cout << "[I] Detection model creating...\n";
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

 //   std::cout << "[I] Key frame extraction engine creating...\n";
    mEngine = mRuntime->deserializeCudaEngine(trtModelStream, size);
    assert(mEngine != nullptr);
    mContext = mEngine->createExecutionContext();
    assert(mContext != nullptr);
    delete[] trtModelStream;

    auto out_dims = mEngine->getBindingDimensions(1);

    mBlob = new float[mInputSize];
    mBoxes = new float[mBoxesSize];
    mScores = new float[mScoresSize];
    mLabels = new float[mLabelsSize];

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(mEngine->getNbBindings() == 4);
    std::cout << "[I] Cuda buffer creating...\n";

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    mInputIndex = mEngine->getBindingIndex("image");

    assert(mEngine->getBindingDataType(mInputIndex) == nvinfer1::DataType::kFLOAT);
    mBoxesIndex = mEngine->getBindingIndex("boxes");
    mScoresIndex = mEngine->getBindingIndex("scores");
    mLabelsIndex = mEngine->getBindingIndex("labels");
    assert(mEngine->getBindingDataType(mBoxesIndex) == nvinfer1::DataType::kFLOAT);
    assert(mEngine->getBindingDataType(mScoresIndex) == nvinfer1::DataType::kFLOAT);
    //    assert(mEngine->getBindingDataType(mLabelsIndex) == nvinfer1::DataType::kFLOAT);
   // int mBatchSize = mEngine->getMaxBatchSize();

    // Create GPU buffers on device
    checkStatus(cudaMalloc(&mBuffers[mInputIndex], mInputSize * sizeof(float)));
    checkStatus(cudaMalloc(&mBuffers[mBoxesIndex], mBoxesSize * sizeof(float)));
    checkStatus(cudaMalloc(&mBuffers[mScoresIndex], mScoresSize * sizeof(float)));
    checkStatus(cudaMalloc(&mBuffers[mLabelsIndex], mLabelsSize * sizeof(float)));

    // Create stream
    std::cout << "[I] Cuda stream creating...\n";
    checkStatus(cudaStreamCreate(&mStream));

 //   std::cout << "[I] Key frame extraction engine created!\n";

    return 0;
}

void StructureInfer::doInference(std::vector<cv::Mat> frames, i_s_map idx_structure_mapping,
    s_frames& structure_frames, f_structures& frame_structures)
{
    double imgH = frames[0].rows;
    double imgW = frames[0].cols;

    mcvOriginSize.width = static_cast<int>(imgW);
    mcvOriginSize.height = static_cast<int>(imgH);

    // 单帧视频结果
    std::vector<cv::Rect> recttemp;
    std::vector<int> labeltemp;
    std::vector<float> probtemp;

    frame_object_boxes frame_object_boxes_i;

    std::cout << "1" << std::endl;
    for (int i = 0; i < frames.size();i++)
    {
        cv::Mat frame = frames[i].clone();
        // cv::imshow("test", frame);
        // cv::waitKey(0);
        cv::Mat drewImg = doSingleInfer(frame, recttemp, labeltemp, probtemp);
        //        rect.push_back(vResults.rect);
        //label.push_back(vResults.label);
       // prob.push_back(vResults.prob);概率框子，标签全都出来了

        frame_object_boxes_i.labels = labeltemp;
        frame_object_boxes_i.probs = probtemp;
        frame_object_boxes_i.rects = recttemp;
        frame_structures[i] = frame_object_boxes_i;
      //  ///  2023.12.26 create by SCJ、update by CHL，核验模型各结构的idx与设定的idx是否一致
       //  int counter = 0;
       //  std::vector<cv::Scalar> rect_color = {
       //    cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255), cv::Scalar(255, 255, 0),
       //    cv::Scalar(128, 192, 255), cv::Scalar(192, 255, 128), cv::Scalar(255, 128, 192),
       //    cv::Scalar(255, 233, 192), cv::Scalar(200, 233, 192), cv::Scalar(100, 200, 50),
       //};//画九种颜色的框子
       //  for (auto& rect : frame_object_boxes_i.rects)
       //{
       //    cv::rectangle(drewImg, rect, rect_color[counter], 2);
       //    std::string label = std::to_string(frame_object_boxes_i.labels[counter]);
       //    cv::putText(drewImg, label, cv::Point(rect.x, rect.y - 20), cv::FONT_HERSHEY_COMPLEX, 1.0f, rect_color[counter], 1);
       //    cv::imshow("test", drewImg);
       //    ++counter;
       //}
       //  cv::waitKey(0);
        for (int j = 0; j < labeltemp.size(); j++)
        {
            // structure_frames[idx_structure_mapping[j]].push_back(i);
            // 此处应该取labeltemp[j], 而非j
            structure_frames[idx_structure_mapping[labeltemp[j]]].push_back(i);
        }
        recttemp.clear();
        labeltemp.clear();
        probtemp.clear();
    }
}

void StructureInfer::preprocess(cv::Mat& src, cv::Mat& dst)
{
    //    mcvOriginSize = src.size();
    cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
    cv::resize(dst, dst, mcvInputSize);
    dst.convertTo(dst, CV_32FC3);
    blobFromImage(dst);//归一化
}

cv::Mat StructureInfer::doSingleInfer(cv::Mat& src, std::vector<cv::Rect>& rect, std::vector<int>& label, std::vector<float>& prob)
{
    cv::Mat originImg = src.clone();
    preprocess(src, src);

    auto start = std::chrono::system_clock::now();
    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    checkStatus(cudaMemcpyAsync(mBuffers[mInputIndex], mBlob, mInputSize * sizeof(float), cudaMemcpyHostToDevice, mStream));
    mContext->enqueueV2(mBuffers, mStream, nullptr);
    checkStatus(cudaMemcpyAsync(mScores, mBuffers[mScoresIndex], mScoresSize * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    checkStatus(cudaMemcpyAsync(mBoxes, mBuffers[mBoxesIndex], mBoxesSize * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    checkStatus(cudaMemcpyAsync(mLabels, mBuffers[mLabelsIndex], mLabelsSize * sizeof(float), cudaMemcpyDeviceToHost, mStream));
    cudaStreamSynchronize(mStream);
    auto end = std::chrono::system_clock::now();

    //std::cout << "[I] Inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 50 << " ms" << std::endl;

    decodeOutputs(rect, label, prob);

    return originImg;
}


void StructureInfer::blobFromImage(cv::Mat& image)
{
    int channels = image.channels();
    int rows = image.rows;
    int cols = image.cols;

    for (int c = 0; c < channels; c++)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                mBlob[c * rows * cols + row * cols + col] = image.at<cv::Vec3f>(row, col)[c] - vMeans[c];
            }
        }
    }
}

void StructureInfer::decodeOutputs(std::vector<cv::Rect>& rect, std::vector<int>& label, std::vector<float>& prob)
{
    int imgW = mcvOriginSize.width;
    int imgH = mcvOriginSize.height;
    object_box vResults;
    std::vector<std::vector<object_box>> vObjects(mobjectNum);
    generate_proposals(0.2f, vObjects);//算目标检测框

    for (auto v : vObjects) {
        qsort_descent_inplace(v);//通过调用快速排序算法，对给定的目标框向量按照概率值进行降序排序
        if (v.size() == 0) {
            continue;
        }
        vResults = v[0];

        int x0 = vResults.rect.x;
        int y0 = vResults.rect.y;
        int x1 = vResults.rect.x + vResults.rect.width;
        int y1 = vResults.rect.y + vResults.rect.height;

        // 裁剪处理，确保目标框在图像范围内
        // clip
        x0 = (std::max)((std::min)(x0, (imgW - 1)), 0);
        y0 = (std::max)((std::min)(y0, (imgH - 1)), 0);
        x1 = (std::max)((std::min)(x1, (imgW - 1)), 0);
        y1 = (std::max)((std::min)(y1, (imgH - 1)), 0);

        vResults.rect.x = x0;
        vResults.rect.y = y0;
        vResults.rect.width = x1 - x0;
        vResults.rect.height = y1 - y0;
        rect.push_back(vResults.rect);
        label.push_back(vResults.label);
        prob.push_back(vResults.prob);
    }
}

void StructureInfer::generate_proposals(float probThresh, std::vector<std::vector<object_box>>& objects)
{
    int imgW = mcvOriginSize.width;
    int imgH = mcvOriginSize.height;
    float rW = static_cast<float>(imgW) / static_cast<float>(mcvInputSize.width);
    float rH = static_cast<float>(imgH) / static_cast<float>(mcvInputSize.height);//算下输入输出比例

    for (int i = 0; i < 8732; i++)
    {
        std::vector<float> singleClassScore(mobjectNum, 0.0f);
        int basePos = i * mobjectNum;
        int baseBoxPos = i * mobjectNum * 4;

        for (int j = 0; j < mobjectNum; j++)
        {
            singleClassScore[j] = mScores[basePos + j];
        }
        auto argMax = argmax(singleClassScore);
        int label = argMax.first;
        float score = argMax.second;

        if (score > probThresh)
        {
            object_box obj;
            int x1 = static_cast<int>(mBoxes[baseBoxPos + 4 * label] * rW);
            //这行代码的逻辑是将目标提议框的左上角 x 坐标计算出来。mBoxes[baseBoxPos + 4 * label] 是目标提议框的左上角 x 坐标（以归一化坐标表示），rW 是宽度的缩放因子。通过将这两个值相乘并转换为整数，得到了在原始图像中的 x1 坐标。
            int y1 = static_cast<int>(mBoxes[baseBoxPos + 4 * label + 1] * rH);
            int x2 = static_cast<int>(mBoxes[baseBoxPos + 4 * label + 2] * rW);
            int y2 = static_cast<int>(mBoxes[baseBoxPos + 4 * label + 3] * rH);
            int w = x2 - x1;
            int h = y2 - y1;

            obj.rect.x = x1;
            obj.rect.y = y1;
            obj.rect.width = w;
            obj.rect.height = h;
            obj.prob = score;
            obj.label = label;
            objects[label].push_back(obj);
        }
    }
}

void StructureInfer::qsort_descent_inplace(std::vector<object_box>& vObjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = vObjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (vObjects[i].prob > p)
            i++;

        while (vObjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(vObjects[i], vObjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(vObjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(vObjects, i, right);
        }
    }
}

void StructureInfer::qsort_descent_inplace(std::vector<object_box>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

std::pair<int, float> StructureInfer::argmax(std::vector<float>& vProb)
{
    std::pair<int, float> result;
    auto iter = std::max_element(vProb.begin(), vProb.end());
    result.first = static_cast<int>(iter - vProb.begin());
    result.second = *iter;

    return result;
}

void StructureInfer::nms_sorted_bboxes(const std::vector<object_box>& vObjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = static_cast<int>(vObjects.size());

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = static_cast<float>(vObjects[i].rect.area());
    }

    for (int i = 0; i < n; i++)
    {
        const object_box& a = vObjects[i];

        int keep = 1;
        for (int isPick : picked)
        {
            const object_box& b = vObjects[isPick];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[isPick] - inter_area;
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}
