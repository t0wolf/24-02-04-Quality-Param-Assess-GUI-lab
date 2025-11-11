#include "tdi_ivs_keypoints_inferer.h"

TDIIVSKeypointsInferer::TDIIVSKeypointsInferer(std::string& sEnginePath)
	:MVEAKeypointsInferer(sEnginePath)
{
	m_inputDims = { 1, 3, 256, 192 };  // {height, width}
	m_outputDims = { 1, 4, 64, 48 };
	m_classes = 4;
}

TDIIVSKeypointsInferer::~TDIIVSKeypointsInferer()
{
    if (m_engine != nullptr)
    {
        delete m_engine;
        m_engine = nullptr;
    }
    if (m_context != nullptr)
    {
        delete m_context;
        m_context = nullptr;
    }
}
