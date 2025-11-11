//
// Created by Ziyon on 2022/7/12.
//
#define __STDC_FORMAT_MACROS
#include "gaininfer.h"
#define random(a,b) (rand()%(b-a)+a)



GainInfer::GainInfer(std::string& trtEnginePath, cv::Size& inputSize, std::vector<float> mMean, std::vector<float> mStd)
	: mEnginePath(trtEnginePath)
	, mcvInputSize(inputSize)
	, mInputSize(1 * 3 * mcvInputSize.height * mcvInputSize.width) // `1` & `3` means batch_size and channels
	, mOutputSize(1 * 2)
{
	// { 114.7748, 107.7354, 99.475 }
	this->mMean = { mMean[0], mMean[1], mMean[2] };
	this->mStd = { mStd[0], mStd[1], mStd[2] };
	this->mClassNum = 2; // 总类别数目
	build();
}


GainInfer::~GainInfer()
{
	delete this->mContext;
	delete this->mRuntime;
	delete this->mEngine;

	cudaStreamDestroy(mStream);

	checkStatus(cudaFree(mBuffers[mInputIndex]));
	checkStatus(cudaFree(mBuffers[mOutputIndex]));

	delete[] this->mBlob;
	delete[] this->mProb;
}



void GainInfer::build() {
	cudaSetDevice(0);
	char *trtModelStream{ nullptr };
	size_t size{ 0 };

	std::ifstream file(mEnginePath, std::ios::binary);
	std::cout << "[I] Gain classification model creating...\n";
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

	std::cout << "[I] Gain classification  engine creating...\n";
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

	std::cout << "[I] Gain classification engine created!\n";

}



void GainInfer::preprocess(cv::Mat& src, cv::Mat& dst)
{
	cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
	cv::resize(dst, dst, mcvInputSize);
	dst.convertTo(dst, CV_32FC3);
	blobFromImage(dst);
}

std::vector<cv::Mat>
GainInfer::videoInput(std::vector<cv::Mat> video)
{
	std::vector<cv::Mat> video_w = video;

	return video_w;
}


void GainInfer::blobFromImage(cv::Mat& image)
{
	long int channels = image.channels();
	long int rows = image.rows;
	long int cols = image.cols;
	long int img_channel_pixs = rows * cols;

	// C,H,W
	for (int c = 0; c < channels; c++)
	{
		for (int row = 0; row < rows; row++)
		{
			for (int col = 0; col < cols; col++)
			{
				mBlob[c * img_channel_pixs + row * cols + col] = (image.at<cv::Vec3f>(row, col)[c]/255.0 - mMean[c])/ mStd[c];
			}
		}
	}
}



std::vector<float> GainInfer::doSingleInfer(cv::Mat& image)
{
	std::vector<float> vProb(this->mClassNum, 0.0f);

	// turn std::vector<cv::Mat> to a float array

	cv::Mat image_r = image.clone();
	preprocess(image, image_r);

	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	auto start = std::chrono::system_clock::now();
	checkStatus(cudaMemcpyAsync(mBuffers[mInputIndex], mBlob, mInputSize * sizeof(float), cudaMemcpyHostToDevice, mStream));
	mContext->enqueueV2(mBuffers, mStream, nullptr);
	checkStatus(cudaMemcpyAsync(mProb, mBuffers[mOutputIndex], mOutputSize * sizeof(float), cudaMemcpyDeviceToHost, mStream));
	cudaStreamSynchronize(mStream);
	auto end = std::chrono::system_clock::now();

	for (int i = 0; i < this->mClassNum; i++)
	{
		vProb[i] = mProb[i];
		//std::cout << mProb[i] << " ";
	}

	return vProb;

}


std::vector<int>
GainInfer::Inference(std::vector<cv::Mat>& frames)
{
	std::vector<std::vector<float>> vProbs;
	for (int i = 0; i < frames.size(); i++)
	{
		cv::Mat frame = frames[i].clone();
		vProbs.push_back(doSingleInfer(frame));
	}

	std::vector<int> finalresult;
	for (int i = 0; i < frames.size(); i++) {
		finalresult.push_back(std::max_element(vProbs[i].begin(), vProbs[i].end()) - vProbs[i].begin());
	}

	return finalresult;
}

void
GainInfer::doInference(std::vector<cv::Mat> video, std::vector<int>& frame_gain_scores)
{
	auto vFrames = videoInput(video);

	frame_gain_scores = Inference(vFrames);
    std::cout << "[I] Inference phase ends.\n";
}


std::pair<int, float> GainInfer::argmax(std::vector<float>& vProb)
{
	std::pair<int, float> result;
	auto iter = std::max_element(vProb.begin(), vProb.end());
	result.first = static_cast<int>(iter - vProb.begin());
	result.second = *iter;

	return result;
}


//int main(int argc, char **argv) {
//	std::vector<float> mMean = {0.5f, 0.5f,0.5f };
//	std::vector<float> mStd = {0.5f, 0.5f,0.5f };
//	std::vector<int> frame_gain_scores;
//	std::string a = "../../extern/gain_classification/psaxgv/gain_psaxgv_bright.engine";
//	cv::Size inputSize(84, 84);
//	GainInfer model(a, inputSize, mMean, mStd);
//	
//	std::string videoPath = "F:\BME_New\Echocardiography_Codes\Quality_Assessment_Online/USm.1.2.840.113663.1500.1.367419684.3.12.20221101.155234.46_SePACS.dcm.avi";  // 待推理视频的路径
//	std::vector<cv::Mat> frames;
//	std::vector<cv::Mat> frames_gray;
//	int totalCount = 0;
//	utils preprocess;
//	double sus = preprocess.getVideodata(videoPath, frames, frames_gray,totalCount);
//	model.doInference(frames, frame_gain_scores);
//	std::cout << "final result is " << frame_gain_scores.size() << std::endl;
//	return 0;
//}