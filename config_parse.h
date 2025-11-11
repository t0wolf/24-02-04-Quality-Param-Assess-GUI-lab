#pragma once
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <QDebug>


class ConfigParse
{
public:
    ConfigParse(std::string configFilePath);

    std::string getEncoderIP();

    std::pair<float, float> getModelWidthDepth();

    YAML::Node getAllNode();

    bool getSpecifiedNode(std::string nodeName, std::string& value);

private:
    YAML::Node m_config;

};
