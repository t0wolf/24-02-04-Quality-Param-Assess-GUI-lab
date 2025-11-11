#include "pv_keypoints_inferer.h"

PVKeypointsInferer::PVKeypointsInferer(std::string& sEnginePath)
	:MVEAKeypointsInferer(sEnginePath)
{
	m_inputDims = { 1, 3, 256, 192 };  // {height, width}
	m_outputDims = { 1, 1, 64, 48 };
	m_classes = 1;
}

PVKeypointsInferer::~PVKeypointsInferer()
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
