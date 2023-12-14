#ifndef YOLO_H
#define YOLO_H
#include "ForwardEngine.h"
#include "Utils.h"

using namespace std;
using namespace cfg;
using namespace postprocess;

struct CutBox {
    uint32_t x1 = 0;
    uint32_t y1 = 0;
    uint32_t x2 = 0;
    uint32_t y2 = 0;
};

class Yolo {
public:
    explicit Yolo(std::string &cfgPath);
    ~Yolo();
    void Preprocess(void* img, vector<uint32_t> &imageShape);
    int32_t Infer(char* img, vector<uint32_t> &imageShape, vector<DetectBbox>& result);
    int32_t MutScalInfer(char* img, vector<uint32_t> &imageShape, CutBox &CropBox, vector<DetectBbox>& result);

    DetectModelDesc modelDesc;
    ot_video_frame input;
    ForwardEngine *engine;
    DetectPostprocess *detectPostprocess;

private:
    void CutYuv(char* SrcImg, char* DstImg,vector<uint32_t> &imageShape, CutBox &CropBox);
    vector<DetectBbox> NMS(vector<DetectBbox>& result);
};
#endif // FORWARD_ENGINE_H