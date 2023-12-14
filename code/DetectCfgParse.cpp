#include <fstream>
#include <map>
#include "DetectCfgParse.h"
#include <cmath>
using namespace cfg;

float deSigmoid(float value)
{
    return -std::log(1.0 / value - 1.0);
}

int32_t DetectCfgParse::CfgParse(string &cfgPath, string &modelPath, string &imagePath, DetectModelDesc &modelDesc)
{
    map<string, string> cfgData;
    bool ret = Utils::File2Map(cfgPath, cfgData);
    if (!ret) {
        ERROR_LOG("Configfile load fail!");
        return FAILURE;
    }
    modelPath = Utils::GetValue(cfgData, "omPath");
    imagePath = Utils::GetValue(cfgData, "imagePath");
    LOG_CHECK_RETURN(Utils::CheckFile(imagePath) != SUCCESS, FAILURE, "Check image path failed");
    LOG_CHECK_RETURN(Utils::CheckFile(modelPath) != SUCCESS, FAILURE, "Check model path failed");

    modelDesc.imageShape = Utils::StringToInt(Utils::SplitString(Utils::GetValue(cfgData, "imageShape"), ","));
    modelDesc.nmsIouThresHold = atof(Utils::GetValue(cfgData, "nmsIouThresHold").c_str());
    modelDesc.inputShape = Utils::StringToInt(Utils::SplitString(Utils::GetValue(cfgData, "inputShape"), ","));
    modelDesc.anchor = Utils::StringToInt(Utils::SplitString(Utils::GetValue(cfgData, "anchor"), ","));
    modelDesc.featmapStrides = Utils::StringToInt(Utils::SplitString(Utils::GetValue(cfgData, "featmapStrides"), ","));
    vector<string> classes = Utils::SplitString(Utils::GetValue(cfgData, "classes"), ",");

    for (uint32_t i = 0; i < classes.size(); i++) {
        vector<string> curClasses = Utils::SplitString(classes[i], ":");
        string key = curClasses[0];
        float val = atof(curClasses[1].c_str());
        modelDesc.totalClassName.push_back(key);
        modelDesc.classConfThres.push_back(val);
        modelDesc.classes[key] = val;
    }
    auto minValueIt = std::min_element(modelDesc.classConfThres.begin(), modelDesc.classConfThres.end());
    modelDesc.deMinConfThres = deSigmoid(*minValueIt);
    
    return SUCCESS;
}