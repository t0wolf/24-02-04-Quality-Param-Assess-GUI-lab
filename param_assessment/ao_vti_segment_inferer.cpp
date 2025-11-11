#include "ao_vti_segment_inferer.h"

AoVTISegmentInferer::AoVTISegmentInferer(std::string& sEngineFilePath)
	//: SegmentInferBase(sEngineFilePath)
{
    m_classes = 2;
    m_outputDims0 = { 1, 37, 8400 };
    m_outputDims1 = { 1, 32, 160, 160 };
    m_inputDims = { 1, 3, 640, 640 };

    std::ifstream engineFile(sEngineFilePath, std::ios::binary);
    if (engineFile.fail())
    {
        return ;
    }

    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger::gLogger.getTRTLogger());
    m_engine = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
    assert(m_engine != nullptr);

    m_context = m_engine->createExecutionContext();
    assert(m_context != nullptr);

    //return ;
}

AoVTISegmentInferer::~AoVTISegmentInferer()
{
    if (m_engine != nullptr)
    {
        delete m_engine;
        m_engine = nullptr;
    }
    
}

int AoVTISegmentInferer::doInference(cv::Mat& src, std::vector<cv::Mat>& vMasks)
{
    int ret = doSingleInfer(src, vMasks);
    return ret;
}

std::vector<cv::Mat> AoVTISegmentInferer::scaleMasks(const std::vector<cv::Mat>& masks, const cv::Size& originalSize) {
    std::vector<cv::Mat> scaledMasks;
    for (const auto& mask : masks) {
        cv::Mat scaledMask;
        cv::resize(mask, scaledMask, originalSize, 0, 0, cv::INTER_LINEAR);
        scaledMasks.push_back(scaledMask);
    }
    return scaledMasks;
}

int AoVTISegmentInferer::doSingleInfer(cv::Mat& src, std::vector<cv::Mat>& vMasks)
{
    m_inputImageIdx = m_engine->getBindingIndex("images");
    m_outputIdx0 = m_engine->getBindingIndex("output0");
    m_outputIdx1 = m_engine->getBindingIndex("output1");
    m_context->setBindingDimensions(m_inputImageIdx, m_inputDims);
    m_context->setBindingDimensions(m_outputIdx0, m_outputDims0);
    m_context->setBindingDimensions(m_outputIdx1, m_outputDims1);
    void* inputMem{ nullptr };
    void* outputMem1{ nullptr };
    void* outputMem0{ nullptr };
    size_t inputSize = getMemorySize(m_inputDims, sizeof(float));
    size_t outputSize0 = getMemorySize(m_outputDims0, sizeof(float));
    size_t outputSize1 = getMemorySize(m_outputDims1, sizeof(float));
    if (cudaMalloc(&inputMem, inputSize) != cudaSuccess)
    {
        logger::gLogError << "ERROR: input cuda memory allocation failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputMem0, outputSize0) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputSize0 << " bytes" << std::endl;
        return 0;
    }
    if (cudaMalloc(&outputMem1, outputSize1) != cudaSuccess)
    {
        logger::gLogError << "ERROR: output cuda memory allocation failed, size = " << outputSize1 << " bytes" << std::endl;
        return 0;
    }
    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: cuda stream creation failed." << std::endl;
        return 0;
    }

    float* blob = new float[getElementNum(m_inputDims)];
    float* output0 = new float[getElementNum(m_outputDims0)];
    float* output1 = new float[getElementNum(m_outputDims1)];
    //cv::imshow("test", src);
    //cv::waitKey(0);
    cv::Mat dst = src.clone();
    preprocess(src, dst, blob);

    if (cudaMemcpyAsync(inputMem, blob, inputSize, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of input failed, size = " << inputSize << " bytes" << std::endl;
        return 0;
    }

    // Run TensorRT inference
    void* bindings[] = { inputMem,outputMem1,outputMem0 };
    //if (m_context == nullptr)
    //{
    //    logger::gLogError << "ERROR: mImpl is nullptr" << std::endl;
    //    return false;
    //}
    bool status = m_context->enqueueV2(bindings, stream, nullptr);
    if (!status)
    {
        logger::gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return 0;
    }

    if (cudaMemcpyAsync(output0, outputMem0, outputSize0, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputSize0 << " bytes" << std::endl;
        return 0;
    }
    if (cudaMemcpyAsync(output1, outputMem1, outputSize1, cudaMemcpyDeviceToHost, stream) != cudaSuccess)
    {
        logger::gLogError << "ERROR: CUDA memory copy of output failed, size = " << outputSize1 << " bytes" << std::endl;
        return 0;
    }
   // postprocess(output0, output1, src);
    //std::vector<cv::Mat> masks;
    //for (int i = 0; i < 32; ++i) {
    //    cv::Mat mask(160, 160, CV_32F, output1 + i * 160 * 160);
    //    masks.push_back(mask);
    //}
    int size_t[3] = { 32,160, 160 };
    cv::Mat mat_t(3, size_t, CV_32FC1,output1);
    for (int i = 0; i < size_t[0]; i++)
    {
        for (int j = 0; j < size_t[1]; j++)
         {
            float* ptr = mat_t.ptr<float>(i,j);
            for (int k = 0; k < size_t[2]; k++)
            {
                ptr[k] = output1[i*size_t[1]* size_t[2]+j*size_t[2]+k];
            }
        }
    }

    std::vector<cv::Mat> detects;
    for (int b = 0; b < 1; ++b) {
        cv::Mat mat(37, 8400, CV_32F, output0 + b * 3 * 8400);
        detects.push_back(mat.clone());  // 拷贝数据，避免原始数据修改影响
    }
    postProcess(detects,mat_t,src, vMasks);

    delete[] blob;
    delete[] output0;
    delete[] output1;
    cudaFree(inputMem);
    cudaFree(outputMem0);
    cudaFree(outputMem1);
    cudaStreamDestroy(stream);
    return 1;
}

int AoVTISegmentInferer::preprocess(cv::Mat& src, cv::Mat& dst, float* blob)
{
    //m_imgOriginSize = src.size();
    cv::cvtColor(dst, dst, cv::COLOR_BGR2RGB);
    cv::resize(dst, dst, cv::Size(640, 640));
    dst.convertTo(dst, CV_32FC3);

    dst = dst / 255.0f;

    int channels = dst.channels();
    int cols = dst.cols;
    int rows = dst.rows;

    for (int c = 0; c < channels; c++)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int col = 0; col < cols; col++)
            {
                blob[c * rows * cols + row * cols + col] = dst.at<cv::Vec3f>(row, col)[c];
            }
        }
    }

    //blobFromImage(dst, blob);
    return 1;
}

void AoVTISegmentInferer::postProcess(std::vector<cv::Mat>& outs,cv::Mat& output1,cv::Mat &frame, std::vector<cv::Mat>& vMasks)
{
    cv::Size outputSize(m_outputDims0.d[2], m_outputDims0.d[3]);
    int num_proposal = outputSize.width; // 25200
    int out_dim2 = 37;  // 
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;    //  opencv里保存box的
    std::vector<std::vector<float>> mask_output0;
    std::vector<int> classIds;  // 后面画图的时候根据id找类别名
    int img_w = frame.cols;
    int img_h = frame.rows;

    float ratioh = (float)frame.rows / 640, ratiow = (float)frame.cols / 640;
    ///xmin,ymin,xamx,ymax,box_score,class_score
    float* pdata = (float*)outs[0].data;  // 定义浮点型指针，

    for (int i = 0; i < num_proposal; ++i) // 遍历所有的num_pre_boxes
    {
        int index = 4 * num_proposal;      // prob[b*num_pred_boxes*(classes+5)]  
        float obj_conf = pdata[index+i ];  // 置信度分数

        if (obj_conf > 0.5)  // 大于阈值
        {
            cv::Mat scores(1, 1, CV_32FC1, pdata + index + 5);     // 这样操作更好理解，定义一个保存所有类别分数的矩阵[1,80]
            cv::Point classIdPoint; //定义点
            double max_class_socre; // 定义一个double类型的变量保存预测中类别分数最大值
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);  // 求每类类别分数最大的值和索引

            max_class_socre = obj_conf;   // 最大的类别分数*置信度
            std::vector<float> mask_list;
            for (int j = 5; j < out_dim2; j++)
            {
                mask_list.push_back(pdata[j * num_proposal + i]);
            }
            float cx = pdata[0 * num_proposal + i];  // x
            float cy = pdata[1 * num_proposal + i];  // y
            float w = pdata[2 * num_proposal + i];   // w
            float h = pdata[3 * num_proposal + i];   // h

            float x0 = float((cx - 0.5 * w) * ratiow);  // *ratiow，变回原图尺寸
            float y0 = float((cy - 0.5 * h) * ratioh);
            float x1 = float((cx + 0.5 * w) * ratiow);
            float y1 = float((cy + 0.5 * h) * ratioh);

            // clip
            x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.0f);
            y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.0f);
            x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.0f);
            y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.0f);

            mask_output0.push_back(mask_list);
            confidences.push_back((float)max_class_socre);
            boxes.push_back(cv::Rect(x0, y0, (int)(x1 - x0), (int)(y1 - y0)));  //（x,y,w,h）
 //               classIds.push_back(class_idx);  // 
            //drawPred(1, boxes[0].x, boxes[0].y,
            //    boxes[0].x + boxes[0].width, boxes[0].y + boxes[0].height, frame, 1);
        }
    }
    // 进行nms和画图
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences,0.5, 0, indices);
    cv::Mat mask_origin = cv::Mat::zeros(frame.size(), CV_32FC1);
    //经过非极大值抑制后，得到对应的索引，对指定1*37，框+索引进行操作
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        std::vector<float> mask_real = mask_output0[idx];//output0的后32位掩码
        cv::Size shape(640, 640);
        //cv::Mat mask_real_mat(mask_real);
        //cv::Mat output1_mat(output1);
        cv::Mat mask_real_mat = cv::Mat(1, mask_real.size(), CV_32FC1, cv::Scalar::all(0.0));
        for (int i = 0; i < mask_real_mat.rows; i++)
        {
            float* ptr = mask_real_mat.ptr<float>(i);			// 获取矩阵第i行首元素的指针，即指向m[i][0]
            for (int j = 0; j < mask_real_mat.cols; j++)
                ptr[j] = mask_real[i * mask_real_mat.cols +j];
        }

        cv::Mat protos_reshaped = output1.reshape(0, 32);
        // 将 masks_in 与 protos_reshaped 进行矩阵乘法
        cv::Mat result = mask_real_mat * protos_reshaped;
        cv::Mat MSAK_sa = result.reshape(0, 160);
        cv::resize(MSAK_sa, MSAK_sa, cv::Size(frame.cols, frame.rows), 0, 0, cv::INTER_LINEAR);//RESIZE到原图
        sigmoidActivation(MSAK_sa);
     //   drawPred(1, box.x, box.y,box.x + box.width, box.y + box.height, frame, 1);

         // 上采样到640x640
            // 创建一个与MSAK_sa大小相同的掩膜图像，初始值为零
        cv::Mat mask = cv::Mat::zeros(MSAK_sa.size(), MSAK_sa.type());

        // 对于每个矩形区域，将掩膜图像对应的区域设为1
        for (const auto& box : boxes) {
            mask(box).setTo(1);
        }

        // 将掩膜应用于原始图像，使得只有矩形区域保留，其他区域置零
       // cv::Mat result;
      //  MSAK_sa.copyTo(result, mask);
        cv::Mat cropped_img = MSAK_sa(cv::Rect(box.x, box.y, box.width, box.height));//那几个波峰
        cv::Mat mask1 = MSAK_sa.clone();
        // 将掩膜图像的指定矩形区域设为1
      //  mask(box);
        mask1.setTo(0);
        cropped_img.copyTo(mask1(cv::Rect(box.x, box.y, box.width, box.height)));
        mask_origin +=  mask1;
    }

    //看一下覆盖到原图的效果
    // 创建一个相同大小的 Mat 存储转换后的数据
    cv::Mat mask_uchar;

    // 归一化到0-255范围
    cv::normalize(mask_origin, mask_origin, 0, 255, cv::NORM_MINMAX);

    // 将float类型转换为unsigned char类型
    mask_origin.convertTo(mask_uchar, CV_8UC1);
    vMasks.push_back(mask_uchar);
    //cv::imshow("Result", mask_uchar);
    //cv::waitKey(0);
}
// 对cv::Mat进行sigmoid激活
void AoVTISegmentInferer::sigmoidActivation(cv::Mat& mat) {
    // 遍历矩阵中的每个元素
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            // 应用sigmoid函数
            mat.at<float>(i, j) = 1.0 / (1.0 + std::exp(-(mat.at<float>(i, j))));
            if (mat.at<float>(i, j) > 0.6) 
            {
                mat.at<float>(i, j) = 1;
            }
            else
            {
                mat.at<float>(i, j) = 0;
            }
           // std::cout << mat.at<float>(i, j) << std::endl;
        }
    }
}
//不一定用得到
void AoVTISegmentInferer::drawPred(float conf, int left, int top, int right, int bottom, cv::Mat& frame, int classid)   // Draw the predicted bounding box
{
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 2);

    // 获取标签文本
    std::string label = cv::format("%.2f", conf); // 可以根据需要添加更多信息，例如类别名称
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);

    // 绘制标签文本背景
    cv::rectangle(frame, cv::Point(left, top - labelSize.height), cv::Point(left + labelSize.width, top + baseLine), cv::Scalar(255, 255, 0), cv::FILLED);

    // 绘制标签文本
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);

    cv::imshow("Predictions", frame);
    cv::waitKey(0);  // 按任意键关闭窗口
}

