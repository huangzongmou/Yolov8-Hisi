#include <iomanip>
#include <map>
#include <string>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <vector>
#include "DetectCfgParse.h"
// #include "ForwardEngine.h"
#include "Yolo.h"
#include "DetectPostprocess.h"

Yolo::Yolo(std::string &cfgPath)
{
    string modelPath = "";
    string imagePath = "";
    
    DetectCfgParse::CfgParse(cfgPath, modelPath,imagePath, modelDesc);
    Utils::CreateImageBuf(modelDesc.inputShape[1], modelDesc.inputShape[0], input, ALIGN_16); 
    std::cout<<modelPath<<std::endl;
    engine =  new ForwardEngine(modelPath);
    engine->PrepareResource((void *)input.virt_addr[0]);
    detectPostprocess = new DetectPostprocess(modelDesc);
}

void Yolo::Preprocess(void* img, vector<uint32_t> &imageShape)
{
    uint32_t channel = 3;
    uint32_t half = 2;

    if (imageShape == modelDesc.inputShape) {
        memcpy((void *)input.virt_addr[0], img, imageShape[0] * imageShape[1] * channel / half);
    } else {
        std::cout<<imageShape[0]<<","<<imageShape[1]<<std::endl;
        ot_video_frame tempImage = {};
        Utils::CreateImageBuf(imageShape[1], imageShape[0], tempImage, ALIGN_16);
        for(uint32_t i=0; i<(imageShape[0]*channel / half); i++)
        {
            void *virt_addr = (char *)(tempImage.virt_addr[0]) + i * tempImage.stride[0];
            void *img_addr = (char *)img + i*imageShape[1];
            memcpy(virt_addr, img_addr, imageShape[1]);
        }
        
        // memcpy((void *)tempImage.virt_addr[0], img, imageShape[0] * imageShape[1] * channel / half);
        Utils::ImageResize(tempImage, input);
        Utils::ReleaseImageBuf(tempImage);
    }
   
}

int32_t Yolo::Infer(char* img, vector<uint32_t> &imageShape, vector<DetectBbox>& result)
{

    Preprocess(img, imageShape);
    LOG_CHECK_RETURN(engine->Forward() != SUCCESS, FAILURE, "Forward failed");
    result = detectPostprocess->PostProcess(engine, imageShape);
    // detectPostprocess->DumpResult(result);
    return 0;
}

int32_t Yolo::MutScalInfer(char* img, vector<uint32_t> &imageShape, CutBox &CropBox, vector<DetectBbox>& results)
{
    vector<DetectBbox> resultsOrg;
    vector<DetectBbox> resultsCut;
    uint32_t channel = 3;
    uint32_t half = 2;

    // 裁剪图片也要对齐
    if ((CropBox.x2-CropBox.x1)%2 != 0)
    {
        if (CropBox.x2 >= imageShape[1])
        {
            CropBox.x1 -= 1;
        }else{
            CropBox.x2 += 1;
        }
    }

    if ((CropBox.y2-CropBox.y1)%2 != 0)
    {
        if (CropBox.y2 >= imageShape[1])
        {
            CropBox.y1 -= 1;
        }else{
            CropBox.y2 += 1;
        }
    }

    uint32_t DstImgLen = (CropBox.x2 - CropBox.x1)*(CropBox.y2 - CropBox.y1)*channel/half;
    char* DstImg = new char[DstImgLen];  
    CutYuv((char*)img, DstImg, imageShape, CropBox);

    vector<uint32_t> CutShape = {(CropBox.y2 - CropBox.y1),(CropBox.x2 - CropBox.x1)};
    
    if(Infer(DstImg, CutShape, resultsCut) == FAILURE)return FAILURE;       // 剪切图片输出
    
   
    if(Infer(img, imageShape, resultsOrg) == FAILURE)return FAILURE;;        // 原始图片输出

    
    for (auto& bbox : resultsCut) {
        bbox.minX += CropBox.x1; 
        bbox.minY += CropBox.y1; 
        bbox.maxX += CropBox.x1;
        bbox.maxY += CropBox.y1;
    }

    
    std::vector<DetectBbox> merged(resultsCut.size() + resultsOrg.size());
    std::copy(resultsCut.begin(), resultsCut.end(), merged.begin());
    std::copy(resultsOrg.begin(), resultsOrg.end(), merged.begin() + resultsCut.size());

    results = detectPostprocess->_NMS(merged);

    delete[] DstImg;

    // detectPostprocess->DumpResult(results);
    // std::cout<<results.size()<<std::endl;
    // std::ofstream outputFile("cut.yuv", std::ios::binary);
    // outputFile.write(DstImg, DstImgLen);   //写裁剪的图片
    // outputFile.close();
    
    
    return 0;
}

// vector<DetectBbox> Yolo::NMS(vector<DetectBbox>& result)
// {
//     detectPostprocess->
// }


void Yolo::CutYuv(char* SrcImg, char* DstImg, vector<uint32_t> &imageShape, CutBox &CropBox)
{
    int32_t Width = CropBox.x2 - CropBox.x1;
    int32_t Height = CropBox.y2 - CropBox.y1;

    char *pSrcUvAddr = SrcImg + imageShape[0]*imageShape[1];
    char *pDstUvAddr = DstImg + Width*Height;
    for(int32_t i = CropBox.y1; i < CropBox.y2; i++)
    {
        for(int32_t j = CropBox.x1; j < CropBox.x2; j++)
        {
            DstImg[(i - CropBox.y1)*Width + (j - CropBox.x1)] = SrcImg[i*imageShape[1] + j];
            pDstUvAddr[((i - CropBox.y1)/2*(Width/2)+(j - CropBox.x1)/2)*2] = pSrcUvAddr[((i/2)*(imageShape[1]/2)+(j/2))*2];
            pDstUvAddr[((i - CropBox.y1)/2*(Width/2)+(j - CropBox.x1)/2)*2+1] = pSrcUvAddr[((i/2)*(imageShape[1]/2)+(j/2))*2+1];
        }
    }
}

Yolo::~Yolo()
{
    Utils::ReleaseImageBuf(input);
    delete engine;
    delete detectPostprocess;
}
