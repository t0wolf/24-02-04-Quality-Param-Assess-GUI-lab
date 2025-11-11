#include "angleinfer_a4c.h"

AngleInferA4C::AngleInferA4C(std::string enginePath, cv::Size inputSize, cv::Size outputSize){

    this->mInputSize = inputSize;
    this->mOutputSize = outputSize;
    this->inputWeight = inputSize.width;
    this->inputHeight = inputSize.height;
    this->outputWeight = outputSize.width;
    this->outputHeight = outputSize.height;

    build(enginePath);
}

AngleInferA4C::~AngleInferA4C()
{
    ;
}


void AngleInferA4C::build(std::string enginePath) {
    std::ifstream engineFile(enginePath, std::ios::binary);
    assert(engineFile);

    engineFile.seekg(0, std::ifstream::end);
    int64_t fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    assert(engineFile);

    SampleUniquePtr<IRuntime> runtime = SampleUniquePtr<IRuntime>(createInferRuntime(logger::gLogger.getTRTLogger()));
    assert(runtime);

    ICudaEngine* engineTRT = runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr);
    assert(engineTRT);

    engine = std::shared_ptr<ICudaEngine>(engineTRT, InferDeleter());
    assert(engine);

    context = SampleUniquePtr<IExecutionContext>(engine->createExecutionContext());
    assert(context);

}


std::vector<cv::Point> AngleInferA4C::findMaxRegion(std::vector<std::vector<cv::Point>> contours) {
    int temp[2];
    temp[0] = int(contours[0].size());
    temp[1] = 0;
    for (int i = 1; i < int(contours.size()); i++) {
        if (int(contours[i].size()) > temp[0]) {
            temp[0] = int(contours[i].size());
            temp[1] = i;
        }
    }
    /*for (int i = 1; i < int(contours.size()); i++) {
        if (i != temp[1]) {
            vector<vector<Point>> finalContours;
            finalContours.push_back(contours[i]);
            cv::fillPoly(a, finalContours, 0);
        }
    }*/
    return contours[temp[1]];
}

std::vector<int> AngleInferA4C::findPeaks(std::vector<float> num, int count)
{
    std::vector<int> sign;
    for (int i = 1; i < count; i++)
    {
        /*?????????
         *С??0????-1
         *????0????1
         *????0????0
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
    //???sign????λ????
    //???漫??????С???λ??
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

cv::Point AngleInferA4C::postprocess_img(cv::Mat img) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    std::vector<cv::Point> finalcontour;
    cv::findContours(img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);//??????
    finalcontour = findMaxRegion(contours);
    // // ?????????????????
    cv::Moments mu;
    mu = moments(finalcontour, false);
    cv::Point2f mc;
    mc = cv::Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);
    // ???????????????????

    std::vector<float> dis;
    float temp;
    for (int k = 0; k < int(finalcontour.size()); k++) {
        temp = sqrt(pow(finalcontour[k].x - mc.x, 2) + pow(finalcontour[k].y - mc.y, 2));
        dis.push_back(temp);
    }
    std::vector<int> finalPointy;
    std::vector<cv::Point> finalPoint;
    // ?????
    std::vector<int> finalindex;
    finalindex = findPeaks(dis, dis.size());//??????????????????
    for (int i = 0; i < finalindex.size(); i++) {
        finalPointy.push_back(finalcontour[finalindex[i]].y);
        finalPoint.push_back(finalcontour[finalindex[i]]);
    }
    // ??????????????????
     // ??y????????????
    std::vector<int> reve = finalPointy;
    sort(reve.begin(), reve.end());
    std::vector<int>::iterator fir = find(finalPointy.begin(), finalPointy.end(), reve[0]);

    std::vector<int>::iterator sec = find(fir, finalPointy.end(), reve[0]);
    // ??????????????????
    if (sec != finalPointy.end()) {
        sec = find(finalPointy.begin(), finalPointy.end(), reve[1]);
    }
    // ?????????????
    cv::Point fpoint;
    fpoint.y = reve[0];
    int a = fir - finalPointy.begin();
    int b = sec - finalPointy.begin();
    fpoint.x = (finalPoint[a].x + finalPoint[b].x) / 2;
    return fpoint;
}


cv::Mat AngleInferA4C::preprocess_img(cv::Mat& img) {
    // ?????????????????
    w = scale * img.cols;
    h = scale * img.rows;

    // ?????ü??????????????
    x = (inputWeight - w) / 2;
    y = (inputHeight - h) / 2;

    // ????CV_32FC3?????Mat??????????????С
    cv::Mat re(h, w, CV_32FC3);

    // ?????????С
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);

    // ????????????????CV_32FC3?????Mat????????????
    cv::Mat out(inputHeight, inputWeight, CV_32FC3, cv::Scalar(128, 128, 128));

    // ??????????????????????????????Mat??
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    return out;
}

cv::Point AngleInferA4C::doInference(cv::Mat image) {
    // ????????????????
    BufferManager buffers(engine);

    if (image.empty())
        std::cout << "hehe" << std::endl;
    // ?????????????????????????
    scale = std::min(inputWeight / (image.cols*1.0), inputHeight / (image.rows*1.0));
    cv::Mat img = preprocess_img(image);

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(engine->getBindingName(0)));
    int k = 0;
    for (int i = 0; i < inputWeight; i++) {
        for (int j = 0; j < inputHeight; j++) {
            cv::Vec3f uc_pixel = img.at<cv::Vec3f>(i, j);
            //float aaaaaaa = float(uc_pixel[0]) / 255.0f;
            hostDataBuffer[k] = float(uc_pixel[2]) / 255.0f;
            hostDataBuffer[k + inputWeight * inputHeight] = float(uc_pixel[0]) / 255.0f;
            hostDataBuffer[k + 2 * inputWeight*inputHeight] = float(uc_pixel[1]) / 255.0f; // ????????????????????????????
            k++;
        }
    }
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    assert(status);
    buffers.copyOutputToHost();

    float *output;
    // ??????????
    //region
    output = static_cast<float*>(buffers.getHostBuffer(engine->getBindingName(1)));

    cv::Mat region(cv::Size(w, h), CV_8U);
    float t1, t2;;
    // ???????????
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            t1 = float(output[y * outputHeight + x + i + j * outputWeight]);
            t2 = float(output[y * outputHeight + x + i + j * outputWeight + inputWeight * inputHeight]);
            if (t1 > t2) {
                region.at<uchar>(j, i) = 0.0;
            }
            else {
                region.at<uchar>(j, i) = 255.0;
            }
        }
    }
    // ????CV_32FC3?????Mat???????????????С
    cv::Mat re(image.rows, image.cols, CV_32FC3);
    cv::resize(region, re, re.size(), 0, 0, cv::INTER_CUBIC);//
    cv::Point finalpoint = postprocess_img(re);
    //cv::circle(image, final, 3, 255, -1);
    //namedWindow("aa", WINDOW_NORMAL);
    //imshow("aa", image);

    //std::cout << "finish" << endl;
    return finalpoint;
}


//int main(int argc, char **argv) {
//	OnnxSampleParams params;
//	params.dataDirs.push_back("");
//	params.dataDirs.push_back("G://zzy2//Drum//object_detection//SSD_v2//demo_results//LV");
//	uNetInfer model(params, 256, 256, 256, 256);
//	model.build("G:\\zzy2\\Drum\\application\\model\\gpu\\unet\\unet_8.0GA.engine");
//	clock_t start = clock();
//	Mat a;
//	string filename = "neg_img_266.jpg";
//	a = model.infer(filename);
//
//	namedWindow("img", WINDOW_NORMAL);
//	imshow("img", a);
//	waitKey(0);
//	//imwrite("G://zzy2//Drum//object_detection//SSD_v2//demo_results//LV\\region01.png", a);
//	clock_t finish = clock();
//}
