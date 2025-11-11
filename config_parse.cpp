#include "config_parse.h"

ConfigParse::ConfigParse(std::string configFilePath)
{
    qDebug() << "[I] Parsing Configuration file...";
    try
    {
        m_config = YAML::LoadFile(configFilePath);
        // encoderIP = config["ENCODER_IP"].as<std::string>();
        // std::cout << encoderIP << std::endl;
    }
    catch (const YAML::Exception& e)
    {
        qDebug() << "[E] Config file cannot be loaded: " << e.what();
    }
}

std::string ConfigParse::getEncoderIP()
{
    return m_config["ENCODER_IP"].as<std::string>();
}

std::pair<float, float> ConfigParse::getModelWidthDepth()
{
    return std::make_pair(m_config["MODEL_WIDTH"].as<float>(), m_config["MODEL_DEPTH"].as<float>());
}

YAML::Node ConfigParse::getAllNode()
{
    return m_config;
}

bool ConfigParse::getSpecifiedNode(std::string nodeName, std::string& value)
{
    try
    {
        if (m_config[nodeName])
        {
            value = m_config[nodeName].as<std::string>();
        }
        else
        {
            std::cerr << "[E] Key " << nodeName << " does not exist.\n";
            return false;
        }
    }
    catch (const YAML::Exception& e)
    {
        std::cerr << "[E] Failed to get value " << e.what() << std::endl;
        return false;
    }

    return true;
}
