#pragma once

#include "mvea_keypoints_inferer.h"

class PVKeypointsInferer :public MVEAKeypointsInferer
{
public:
	PVKeypointsInferer(std::string& sEnginePath);

	~PVKeypointsInferer();

};