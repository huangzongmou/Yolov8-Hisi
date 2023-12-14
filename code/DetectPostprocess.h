#ifndef DETECT_POSTPROCESS_H
#define DETECT_POSTPROCESS_H

#include "Utils.h"
#include "ForwardEngine.h"

namespace postprocess {
struct DetectModelDesc {
    float nmsIouThresHold = 0;
    vector<uint32_t> inputShape;
    vector<uint32_t> anchor;
    vector<uint32_t> imageShape;
    vector<uint32_t> featmapStrides;
    map<string, float> classes;
    vector<float> classConfThres;
    float deMinConfThres;                 //反算值
    vector<string> totalClassName;
};

struct DetectBbox {
    float minX = 0;
    float minY = 0;
    float maxX = 0;
    float maxY = 0;
    float confidence = 0;
    uint32_t classIdx = 0;
    string className = "";
    uint32_t mask = 0;
    map<string, float> similarCl;  //分数相近的类别
};

class DetectPostprocess {
public:
    DetectModelDesc m_modelDesc;

    explicit DetectPostprocess(const DetectModelDesc &modelDesc);
    ~DetectPostprocess();
    vector<DetectBbox> PostProcess(ForwardEngine *forwardEngine, vector<uint32_t> &imageShape);
    // vector<DetectBbox> PostProcess(ForwardEngine &forwardEngine, vector<uint32_t> &imageShape);
    void DumpResult(vector<DetectBbox> &result);
    vector<DetectBbox> _NMS(vector<DetectBbox>& result);

private:
    
    vector<uint32_t> m_gridNumHeight;
    vector<uint32_t> m_gridNumWidth;

    float GetBboxMaxClassConfidence(const float *boxValue, uint32_t num, uint32_t &maxIndex) const;
    map<string, float> GetBboxSimilarClassConfidence(float classScore ,float objectScore, const float *boxValue, uint32_t num,uint32_t maxIndex);
    bool IsBboxConfInRange(uint32_t classIdx, float conf) const;
    virtual int32_t CalcBboxInfo(const float *detectResult, vector<DetectBbox> &result,
        uint32_t outputIndex, size_t strideElemNum) const;
    float CalcIou(const DetectBbox &detectBbox1, const DetectBbox &detectBbox2) const;
    void NMS(vector<DetectBbox> &result, float nmsThresh) const;
    static bool BboxConfDescend(const DetectBbox &detectBbox1, const DetectBbox &detectBbox2)
    {
        return detectBbox1.confidence > detectBbox2.confidence;
    }
};
}
#endif // DETECT_POSTPROCESS_H
