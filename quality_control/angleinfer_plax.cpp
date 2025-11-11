#include "angleinfer_plax.h"

AngleInferPLAX::AngleInferPLAX(std::string enginePath, cv::Size inputSize, cv::Size outputSize) {

	this->mInputSize = inputSize;
	this->mOutputSize = outputSize;
	this->inputWeight = inputSize.width;
	this->inputHeight = inputSize.height;
	this->outputWeight = outputSize.width;
	this->outputHeight = outputSize.height;

	build(enginePath);
}

AngleInferPLAX::~AngleInferPLAX()
{
	;
}


void AngleInferPLAX::build(std::string enginePath) {
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

cv::Mat AngleInferPLAX::preprocess_img(cv::Mat& img) {

	w = scale * img.cols;
	h = scale * img.rows;
	x = (inputWeight - w) / 2;
	y = (inputHeight - h) / 2;
	cv::Mat re(h, w, CV_32FC3);
	cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
	cv::Mat out(inputHeight, inputWeight, CV_32FC3, cv::Scalar(128, 128, 128));
	re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
	return out;
}


int AngleInferPLAX::doInference(cv::Mat image) {

	BufferManager buffers(engine);
	if (image.empty())
		std::cout << "Image is empty!" << std::endl;
	scale = std::min(inputWeight / (image.cols * 1.0), inputHeight / (image.rows * 1.0));
	cv::Mat img = preprocess_img(image);

	float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(engine->getBindingName(0)));
	int k = 0;
	for (int i = 0; i < inputWeight; i++) {
		for (int j = 0; j < inputHeight; j++) {
			cv::Vec3f uc_pixel = img.at<cv::Vec3f>(i, j);
			hostDataBuffer[k] = float(uc_pixel[2]) / 255.0f;
			hostDataBuffer[k + inputWeight * inputHeight] = float(uc_pixel[0]) / 255.0f;
			hostDataBuffer[k + 2 * inputWeight * inputHeight] = float(uc_pixel[1]) / 255.0f;
			k++;
		}
	}
	buffers.copyInputToDevice();

	bool status = context->executeV2(buffers.getDeviceBindings().data());
	assert(status);
	buffers.copyOutputToHost();

	float* output;

	//region
	output = static_cast<float*>(buffers.getHostBuffer(engine->getBindingName(1)));

	int result = output[1] > output[0] ? 1.0 : 0.0;

	//cv::circle(image, final, 3, 255, -1);
	//namedWindow("aa", WINDOW_NORMAL);
	//imshow("aa", image);

	//std::cout << "finish" << endl;
	return result;
}

//int main(int argc, char **argv) {
//	Resnet50Infer model(params, 256, 256, 256, 256);
//}