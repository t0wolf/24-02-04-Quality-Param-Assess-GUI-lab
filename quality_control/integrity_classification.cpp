#include "integrity_classification.h"


IntegrityClassification::IntegrityClassification(std::string& enginePath)
	: mEnginePath(enginePath)
	, mInputSize(32 * 3 * 320 * 320)
	, mOutputSize(1 * 4)
	, mClipLen(32)
{
	initialize();
}

IntegrityClassification::~IntegrityClassification()
{
	delete mContext;
	delete mEngine;
	delete mRuntime;

	delete[] mBlob;
	delete[] mProb;
}

void IntegrityClassification::initialize()
{
	cudaSetDevice(0);
	char* trtModelStream{ nullptr };
	size_t size{ 0 };

	std::ifstream file(mEnginePath, std::ios::binary);
	std::cout << "[I] Classification model creating...\n";
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

	mRuntime = nvinfer1::createInferRuntime(mGLogger);
	assert(mRuntime != nullptr);

	std::cout << "[I] Classification engine creating...\n";
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
	mInputIndex = mEngine->getBindingIndex("image");
	assert(mEngine->getBindingDataType(mInputIndex) == nvinfer1::DataType::kFLOAT);

	mOutputIndex = mEngine->getBindingIndex("label");
	assert(mEngine->getBindingDataType(mOutputIndex) == nvinfer1::DataType::kFLOAT);

	// Create GPU buffers on device
	checkStatus(cudaMalloc(&mBuffers[mInputIndex], mInputSize * sizeof(float)));
	checkStatus(cudaMalloc(&mBuffers[mOutputIndex], mOutputSize * sizeof(float)));

	// Create stream
	std::cout << "[I] Cuda stream creating...\n";
	checkStatus(cudaStreamCreate(&mStream));

	std::cout << "[I] Detection engine created!\n";
}

int Find_Mod(std::vector<int>& theta)
{
	std::vector<int>theta_(theta);
	int n = theta.size();
	std::sort(theta_.begin(), theta_.end());
	int i = 0;
	int MaxCount = 1;
	int index = 0;

	while (i < n)//遍历
	{
		int count = 1;
		int j;
		for (j = i; j < n - 1; j++)
		{
			if (theta[j] == theta[j + 1])//存在连续两个数相等，则众数+1
			{
				count++;
			}
			else
			{
				break;
			}
		}
		if (MaxCount < count)
		{
			MaxCount = count;//当前最大众数
			index = j;//当前众数标记位置
		}
		++j;
		i = j;//位置后移到下一个未出现的数字
	}
	std::cout << theta[index] << " " << MaxCount << std::endl;
	return theta[index];
}


int IntegrityClassification::inference(std::vector<cv::Mat>& frames)
{
	int totalFrame = frames.size();
	assert(totalFrame > 0);
	//std::vector<std::vector<int>> vInds;
	//sampleFrame(frames);
	std::vector<cv::Mat> vSingleMats;
	std::vector<int> all_category_results;
	
	int startIndex = 0;
	while (startIndex + mClipLen - 1 < totalFrame) {
		for (int i = 0; i < mClipLen; i++)
		{
			cv::Mat temp = frames[startIndex + i];
			preprocess(temp);
			vSingleMats.push_back(temp);
		}
		assert(vSingleMats.size() == mClipLen);
		auto singleRes = singleInfer(vSingleMats);
		//if (singleRes.second > 0.6f) {
		//	all_category_results.push_back(singleRes.first);
		//}
		all_category_results.push_back(singleRes.first);
		vSingleMats.clear();
		startIndex += mClipLen;
	}
// 处理剩余未达32帧的帧
	if (totalFrame != startIndex) {
		int paddingNum = totalFrame - startIndex - mClipLen;
		while (paddingNum < 0)
		{
			cv::Mat temp = frames[startIndex];
			preprocess(temp);
			vSingleMats.push_back(temp);
			paddingNum++;
		}
		for (int i = startIndex; i < totalFrame; i++) {
			cv::Mat temp = frames[startIndex];
			preprocess(temp);
			vSingleMats.push_back(temp);
		}
		assert(vSingleMats.size() == mClipLen);
		auto singleRes = singleInfer(vSingleMats);
		//if (singleRes.second > 0.6f || all_category_results.empty()) {
		//	all_category_results.push_back(singleRes.first);
		//}
		all_category_results.push_back(singleRes.first);
		vSingleMats.clear();
	}
	//int minValue = Find_Mod(all_category_results);
	int minValue = *std::min_element(all_category_results.begin(), all_category_results.end());
	return minValue;

	/*for (auto vInd : vInds)
	{
		std::vector<cv::Mat> vSingleMats;

		for (int ind : vInd)
		{
			if (ind < frames.size())
			{
				cv::Mat tempMat = frames[ind];
				preprocess(tempMat);
				vSingleMats.push_back(tempMat);
			}

			else
			{
				cv::Mat tempMat = cv::Mat::zeros(cv::Size(256, 256), CV_32FC3);
				preprocess(tempMat);
				vSingleMats.push_back(tempMat);
			}

		}

		singleInfer(vSingleMats);
	}*/

}

void IntegrityClassification::sampleFrame(int totalFrame, std::vector<std::vector<int>>& vInds)
{
	int numClip = 0;
	if (totalFrame < mClipLen * 2)
		numClip = 1;

	else
		numClip = std::ceil(totalFrame / mClipLen / 2);

	vInds.resize(numClip);

	for (int i = 0; i < numClip; i++)
	{
		auto& vInd = vInds[i];
		vInd.resize(mClipLen);

		for (int j = 0; j < mClipLen; j++)
		{
			vInd[j] = (j + i * mClipLen) * 2;
		}
	}
}

void IntegrityClassification::sampleFrame(std::vector<cv::Mat>& vMats)
{
	int totalFrame = vMats.size();
	int padding = 0;

	if (totalFrame < mClipLen)
	{
		padding = mClipLen - totalFrame;
	}

	if (padding)
	{
		cv::Mat lastFrame = vMats.back();
		for (int i = 0; i < padding; i++)
		{
			vMats.push_back(lastFrame);
		}
	}
}

void IntegrityClassification::videoInput(cv::VideoCapture& cap, std::vector<cv::Mat>& vMats)
{
	cv::Mat frame;
	while (cap.read(frame))
	{
		vMats.push_back(frame);
	}
}

void IntegrityClassification::preprocess(cv::Mat& mat)
{
	cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
	cv::resize(mat, mat, cv::Size(320, 320));
	mat.convertTo(mat, CV_32FC3);
}

std::pair<int, float> IntegrityClassification::singleInfer(std::vector<cv::Mat>& vSingleMats)
{
	blobFromVideo(vSingleMats);
	auto start = std::chrono::system_clock::now();
	checkStatus(cudaMemcpyAsync(mBuffers[mInputIndex], mBlob, mInputSize * sizeof(float), cudaMemcpyHostToDevice, mStream));
	mContext->enqueueV2(mBuffers, mStream, nullptr);
	checkStatus(cudaMemcpyAsync(mProb, mBuffers[mOutputIndex], mOutputSize * sizeof(float), cudaMemcpyDeviceToHost, mStream));
	cudaStreamSynchronize(mStream);
	auto end = std::chrono::system_clock::now();

	std::vector<float> vProbs(4, 0.0f);
	for (int k = 0; k < 4; k++)
	{
		vProbs[k] = mProb[k];
	}

	std::pair<int, float> results = argmax(vProbs);

	/*for (int k = 0; k < 2; k++)
		std::cout << mProb[k] << " ";*/

	std::cout << "[I] Inference latency: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
	std::cout << "[I] Classification result: " << results.first << ", probability: " << results.second << std::endl;

	return results;
}

void IntegrityClassification::blobFromVideo(std::vector<cv::Mat>& vSingleMats)
{
	int counter = 0;

	// transpose Mat data to float array.
	for (auto& mat : vSingleMats)
	{
		int row = mat.rows;
		int col = mat.cols;
		int channel = mat.channels();
		for (int c = 0; c < channel; c++)
		{
			for (int i = 0; i < row; i++)
			{
				for (int j = 0; j < col; j++)
				{
					mBlob[counter * row * col * channel + c * row * col + i * col + j] = mat.at<cv::Vec3f>(i, j)[c] - mMeans[c];
				}
			}
		}
		++counter;
	}
}