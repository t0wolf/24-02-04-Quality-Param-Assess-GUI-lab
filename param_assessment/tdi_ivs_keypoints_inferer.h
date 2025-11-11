#pragma once

#include "mvea_keypoints_inferer.h"

class TDIIVSKeypointsInferer :public MVEAKeypointsInferer
{
public:
	TDIIVSKeypointsInferer(std::string& sEnginePath);

	~TDIIVSKeypointsInferer();

};