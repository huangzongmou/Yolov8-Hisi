#include <iostream>
#include <string>
#include "Utils.h"
#include "DetectCfgParse.h"
#include "ForwardEngine.h"
#include "DetectPostprocess.h"
#include "Yolo.h"

using namespace std;
using namespace cfg;
using namespace postprocess;

int32_t main()
{
    string cfgPath = "../data/model.cfg";
    string imagePath = "../data/sample_input.yuv";

    FILE *file = fopen(imagePath.c_str(), "rb");
    if (file == NULL) {
        perror("无法打开文件");
        return 1;
    }
    // 获取文件大小
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    // 分配内存并读取文件内容
    char *buffer = (char *)malloc(fileSize);
    if (buffer == NULL) {
        perror("无法分配内存");
        fclose(file);
        return 1;
    }
    size_t countRead = fread(buffer, 1, fileSize, file);
    if(countRead)
    {
        printf("读取成功\n");
    }
    fclose(file);

    vector<uint32_t> imageShape = {384, 640};
    vector<DetectBbox> results;

    Yolo yolo = Yolo(cfgPath);
    uint32_t ret = yolo.Infer(buffer, imageShape, results);

    if(ret == FAILURE)
    {
        std::cout<<"推理失败"<<std::endl;
    }

    std::cout << results.size() << std::endl;

       for(auto& box :results)
    {

        std::cout<<box.className<<":"<<box.confidence<<std::endl;
        printf("box:%f,%f,%f,%f\n",box.minX,box.minY,box.maxX,box.maxY);
 
    }
    return SUCCESS;
}