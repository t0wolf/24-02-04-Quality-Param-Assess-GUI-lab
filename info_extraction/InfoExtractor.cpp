#include "InfoExtractor.h"

InfoExtractor::InfoExtractor()
	: m_ocrInferer(new OCRInferer("D:/Resources/20240221/ocr_models/ch_det.engine",
		"D:/Resources/20240221/ocr_models/en_rec.engine", "D:/Resources/20240221/en_dict.txt"))
{
	qInfo() << "[I] Info Extractor Initialized.";
}

InfoExtractor::~InfoExtractor()
{
	if (m_ocrInferer != nullptr)
	{
		delete m_ocrInferer;
		m_ocrInferer = nullptr;
	}
}

int InfoExtractor::doInfoExtract(cv::Mat& src, ScaleInfo& scaleInfo, PatientInfo& patientInfo, ModeInfo& modeInfo)
{
	doTextRec(src);
	if (m_vTotalTextInfos.empty())
		return 0;

	doPatientInfoExtract(patientInfo);
	doModeInfoExtract(modeInfo);
	colorModeJudge(modeInfo);

	if (modeInfo.mode == "B-Mode")
		doScaleExtract(scaleInfo);
	//else if (modeInfo.mode == "Doppler-Mode")
	//	specInfoExtract(scaleInfo);

	m_vScaleInfos.clear();
	m_vTotalTextInfos.clear();
	return 1;
}

int InfoExtractor::doInfoExtract(cv::Mat& src, RoIScaleInfo& roiScaleInfo, ScaleInfo& scaleInfo, PatientInfo& patientInfo, ModeInfo& modeInfo)
{
	roiScaleInfo.clear();
	scaleInfo.clear();
	modeInfo.clear();

	doTextRec(src);
	if (m_vTotalTextInfos.empty())
		return 0;

	doPatientInfoExtract(patientInfo);
	doModeInfoExtract(modeInfo);
	colorModeJudge(modeInfo);

	if (!roiScaleInfo.specScaleRect.empty())
		modeInfo.mode = "Doppler-Mode";

	if (modeInfo.mode == "B-Mode")
	{
		doScaleExtract(scaleInfo);
	}
	else if (modeInfo.mode == "Doppler-Mode")
	{
		if (roiScaleInfo.specScaleRect.empty())
		{
			/*specInfoExtract(scaleInfo);*/
			m_vScaleInfos.clear();
			m_vTotalTextInfos.clear();
			return 0;
		}
		else
		{
			m_vTotalTextInfos.clear();
			cv::Mat scaleRectCropped = src(roiScaleInfo.specScaleRect).clone();
			cv::Mat outputImg;
			//equalizeHistogram(scaleRectCropped, scaleRectCropped);
			enhanceContrastBrightness(scaleRectCropped, outputImg, 1.5, -10);
			//cv::Mat drawFrame = src.clone();
			//cv::rectangle(drawFrame, roiScaleInfo.roiRect, cv::Scalar(0, 255, 0), 2);
			//cv::rectangle(drawFrame, roiScaleInfo.specScaleRect, cv::Scalar(0, 255, 0), 2);
			//cv::imshow("spec scale", drawFrame);
			//cv::waitKey(1);
			doTextRec(outputImg);
			if (!m_vTotalTextInfos.empty())
			{
				for (auto& textInfo : m_vTotalTextInfos)
					qDebug() << QString::fromStdString(textInfo.text);
				specInfoExtract(scaleInfo);
			}

		}
	}

	m_vScaleInfos.clear();
	m_vTotalTextInfos.clear();
	return 1;
}

int InfoExtractor::doTextRec(cv::Mat& src)
{
	m_ocrInferer->doInference(src, m_vTotalTextInfos);

	return 1;
}

int InfoExtractor::depthDirectRec()
{
	std::regex length_regex(R"(\b(\d+(\.\d+)?)[ ]?(cm|m)\b)");

	for (auto& textInfo : m_vTotalTextInfos)
	{
		std::string currText = textInfo.text;
		auto words_begin = std::sregex_iterator(currText.begin(), currText.end(), length_regex);
		auto words_end = std::sregex_iterator();

		for (std::sregex_iterator it = words_begin; it != words_end; ++it)
		{
			std::smatch match = *it;
			float fDistInfo = std::stof(match[1].str());
			std::string sUnit = match[3].str();

			if (sUnit == "cm" || sUnit == "m")
			{
				if (std::floor(fDistInfo) == fDistInfo)  // avoid 16.1 cm
				{
					textInfo.distInfo = fDistInfo;
					textInfo.unit = sUnit;

					m_vScaleInfos.push_back(textInfo);
				}
			}
		}
	}

	return 1;
}

int InfoExtractor::specInfoExtract(ScaleInfo& scaleInfo)
{
	std::string unit = "";
	std::vector<std::pair<float, cv::Rect>> scaleValues;
	int unitPositionIdx = specUnitRec(unit);

	scaleInfo.unit = unit;

	specScaleRec(unitPositionIdx, scaleInfo, scaleValues);

	//if (!scaleValues.empty())
	//{
	//	auto currScaleValue = scaleValues[0];
	//	float valueRange = currScaleValue.first;
	//	int pixelRange = currScaleValue.second.y - scaleInfo.unitPositionY;
	//	if (valueRange != 0)
	//	{
	//		scaleInfo.fPixelPerUnit = std::abs(pixelRange / valueRange);
	//	}
	//}
	if (scaleValues.size() >= 3)
	{
		int valueSize = scaleValues.size();
		int midIdx = scaleValues.size() / 2;
		int nextIdx = std::min(midIdx + 1, valueSize - 1);
		int prevIdx = std::max(midIdx - 1, 0);

		float valueRange = scaleValues[nextIdx].first - scaleValues[prevIdx].first;
		int pixelRange = scaleValues[nextIdx].second.y - scaleValues[prevIdx].second.y;
		if (valueRange != 0)
		{
			scaleInfo.fPixelPerUnit = std::abs(pixelRange / valueRange);
			scaleInfo.fSpecScaleRange = valueRange;
		}
	}
	else if (scaleValues.size() >= 2)
	{
		auto& first = scaleValues.front();
		auto& last = scaleValues.back();
		float valueRange = last.first - first.first;
		int pixelRange = last.second.y - first.second.y;
		if (valueRange != 0)
		{
			scaleInfo.fPixelPerUnit = std::abs(pixelRange / valueRange);
			scaleInfo.fSpecScaleRange = valueRange;
		}
	}

	else if (scaleValues.size() == 1)
	{
		auto currScaleValue = scaleValues[0];
		float valueRange = currScaleValue.first;
		int pixelRange = currScaleValue.second.y - scaleInfo.unitPositionY;
		if (valueRange != 0)
		{
			scaleInfo.fPixelPerUnit = std::abs(pixelRange / valueRange);
			scaleInfo.fSpecScaleRange = valueRange;
		}
	}

	else return 0;

	return 1;
}

int InfoExtractor::specScaleRec(int unitPositionIdx, ScaleInfo& scaleInfo, std::vector<std::pair<float, cv::Rect>>& scaleValues)
{
	cv::Rect unitPosition = m_vTotalTextInfos[unitPositionIdx].boundingBox;
	scaleInfo.unitPositionY = unitPosition.y + unitPosition.height / 2;

	std::regex scaleRegex("(?:\\-\\s*)?(-?\\d+(\\.\\d+)?)");

	for (size_t i = 0; i < m_vTotalTextInfos.size(); ++i) 
	{
		if (i != unitPositionIdx) 
		{
			// 确认文本x坐标接近单位的x坐标
			//if (std::abs(m_vTotalTextInfos[i].boundingBox.x - unitPosition.x) < 20) 
			//{
				std::smatch matchResults;
				if (std::regex_search(m_vTotalTextInfos[i].text, matchResults, scaleRegex)) 
				{
					try
					{
						float value = std::stof(matchResults[1]);
						if (matchResults[0].matched)
							value = -value;
						scaleValues.emplace_back(value, m_vTotalTextInfos[i].boundingBox);
					}
					catch (const std::invalid_argument& ia) {
						// 如果转换失败，则忽略该文本
					}
				}
			//}
		}
	}

	std::sort(scaleValues.begin(), scaleValues.end(), [](const auto& a, const auto& b) {
		return a.second.y < b.second.y;
	});

	checkScaleResults(unitPositionIdx, scaleValues);
	
	return 1;
}

int InfoExtractor::checkScaleResults(int unitPositionIdx, std::vector<std::pair<float, cv::Rect>>& scaleValues)
{
	cv::Rect unitPosition = m_vTotalTextInfos[unitPositionIdx].boundingBox;
	scaleValues.erase(std::remove_if(scaleValues.begin(), scaleValues.end(), [](const std::pair<float, cv::Rect>& pair) {
		return pair.first == 0.0f;
	}), scaleValues.end());

	for (auto& scaleValue : scaleValues)
	{
		float value = scaleValue.first;
		cv::Rect position = scaleValue.second;

		if (value > 0.0f)
		{
			if (position.y > unitPosition.y)
				value = -value;
		}

		else if (value < 0.0f)
		{
			if (position.y < unitPosition.y)
				value = -value;
		}

		else if (value == 0.0f)
		{

		}

		scaleValue.first = value;
	}

	return 1;
}

int InfoExtractor::specUnitRec(std::string& unit)
{
	//std::vector<std::string> possibleUnits{ "- m/s", "- cm/s", "- mm/s" };
	//std::string pattern = "(-\\s*[a-zA-Z/\\^]+)|(\\[.*?\\])";
	//std::regex unitRegex(pattern);

	// 正则表达式：匹配可能存在OCR误识别的减号
	std::regex unitWithOptionalDashRegex("(?:\\-\\s*)?(cm/s|m/s)");

	// 正则表达式：匹配可能存在左右方括号包裹的速度单位
	std::regex unitWithBracketsRegex("(\\[)?(cm/s|m/s)(\\])?");

	int unitPositionIdx = -1;

	for (int i = 0; i < m_vTotalTextInfos.size(); ++i)
	{
		std::smatch matchResults;

		if (std::regex_search(m_vTotalTextInfos[i].text, matchResults, unitWithOptionalDashRegex)) 
		{
			unit = matchResults[0];
			unitPositionIdx = i;
			break;
		}

		else if (std::regex_search(m_vTotalTextInfos[i].text, matchResults, unitWithBracketsRegex)) 
		{
			unit = matchResults[0];
			unitPositionIdx = i;
			break;
		}
		else 
		{
			
		}
		//if (std::regex_search(m_vTotalTextInfos[i].text, matchResults, unitRegex))
		//{
		//	int matchIdx = 0;
		//	if (matchResults[1].matched)
		//	{
		//		matchIdx = 1;
		//	}
		//	else if (matchResults[2].matched)
		//	{
		//		matchIdx = 2;
		//	}

		//	unit = matchResults[matchIdx];
		//	unitPositionIdx = i;
		//	break;
		//}
		//for (const auto& possibleUnit : possibleUnits)
		//{
		//	if (m_vTotalTextInfos[i].text.find(possibleUnit) != std::string::npos)
		//	{
		//		unit = possibleUnit;
		//		unitPositionIdx = i;
		//		break;
		//	}
		//}

		//if (unitPositionIdx != -1) break;
	}

	if (unitPositionIdx == -1) return 0;

	return unitPositionIdx;
}

int InfoExtractor::parseScaleInfo()
{
	for (auto& textInfo : m_vScaleInfos)
	{

	}
	return 0;
}

int InfoExtractor::doScaleExtract(cv::Mat& src, ScaleInfo& scaleInfo)
{
	doTextRec(src);

	depthDirectRec();

	scaleInfo = getScaleInfos();
	return 1;
}

int InfoExtractor::doScaleExtract(ScaleInfo& scaleInfo)
{
	depthDirectRec();
	scaleInfo = getScaleInfos();
	return 1;
}

int InfoExtractor::doPatientInfoExtract(PatientInfo& patientInfo)
{
	std::regex name_regex(R"(([A-Z][a-z]+),\s*([A-Z][a-z]+))");
	std::regex id_regex(R"(\d{6,12})");
	std::string patientName;
	std::string patientID;

	std::vector<std::string> name_candidates;
	std::vector<std::string> id_candidates;

	for (size_t i = 0; i < m_vTotalTextInfos.size(); ++i) 
	{
		if (std::regex_match(m_vTotalTextInfos[i].text, name_regex)) 
		{
			name_candidates.push_back(m_vTotalTextInfos[i].text);
		}
		else if (std::regex_match(m_vTotalTextInfos[i].text, id_regex)) 
		{
			id_candidates.push_back(m_vTotalTextInfos[i].text);
		}
	}

	int name_min_y = INT_MAX;
	int id_min_y = INT_MAX;

	for (size_t i = 0; i < m_vTotalTextInfos.size(); ++i)
	{
		if (std::find(name_candidates.begin(), name_candidates.end(), m_vTotalTextInfos[i].text) != name_candidates.end() && m_vTotalTextInfos[i].boundingBox.y < name_min_y)
		{
			patientName = m_vTotalTextInfos[i].text;
			name_min_y = m_vTotalTextInfos[i].boundingBox.y;
		}
		else if (std::find(id_candidates.begin(), id_candidates.end(), m_vTotalTextInfos[i].text) != id_candidates.end() && m_vTotalTextInfos[i].boundingBox.y < id_min_y)
		{
			patientID = m_vTotalTextInfos[i].text;
			id_min_y = m_vTotalTextInfos[i].boundingBox.y;
		}
	}

	patientInfo = { patientName, patientID };

	return 1;
}

int InfoExtractor::doModeInfoExtract(ModeInfo& modeInfo)
{
	bool bIsKeywordFound = false;

	for (int i = 0; i < m_vTotalTextInfos.size(); ++i)
	{
		std::string text = m_vTotalTextInfos[i].text;
		for (const auto& keyword : m_dopplerKeywords)
		{
			if (text.find(keyword) != std::string::npos)
			{
				modeInfo.mode = "Doppler-Mode";
				
				if (keyword == "PW" || keyword == "CW")
					modeInfo.specMode = keyword;

				bIsKeywordFound = true;
				break;
			}
			else
			{
				modeInfo.mode = "B-Mode";
			}
		}
		if (bIsKeywordFound)
			break;
	}
	return 1;
}

int InfoExtractor::colorModeJudge(ModeInfo& modeInfo)
{
	bool bIsKeywordFound = false;

	for (int i = 0; i < m_vTotalTextInfos.size(); ++i)
	{
		std::string text = m_vTotalTextInfos[i].text;
		for (const auto& keyword : m_colorKeywords)
		{
			if (text.find(keyword) != std::string::npos)
			{
				modeInfo.bIsColorMode = true;

				bIsKeywordFound = true;
				break;
			}
			else
			{
				modeInfo.bIsColorMode = false;
			}
		}
		if (bIsKeywordFound)
			break;
	}
	return 1;
}
