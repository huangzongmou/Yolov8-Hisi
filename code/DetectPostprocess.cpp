#include <iomanip>
#include <map>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <cfloat>
#include "DetectPostprocess.h"

using namespace postprocess;

namespace {
const float HALF = 0.5;
const int32_t OUTPUT_X1_INDEX = 0;
const int32_t OUTPUT_Y1_INDEX = 1;
const int32_t OUTPUT_X2_INDEX = 2;
const int32_t OUTPUT_Y2_INDEX = 3;
const int32_t OUTPUT_CLASS_INDEX = 4;

const int32_t SCORE_INTERGER_WIDTH = 9;
const int32_t SCORE_DECIMAL_WIDTH = 8;
const int32_t COORD_INTERGER_WIDTH = 4;
const int32_t COORD_DECIMAL_WIDTH = 2;

void PrintResult(const vector<DetectBbox> &result)
{
    if (result.empty()) {
        INFO_LOG("The model does not detect any object in the sample image. "
        "There maybe no object in this image or objects are hard to detect. "
        "Please change sample_input image.");
        return;
    }
    vector<int> clsNum;
    float cId = result[0].classIdx;
    int validNum = 0;
    for (size_t loop = 0; loop < result.size(); loop++) {
        if (result[loop].classIdx == cId) {
            validNum++;
        } else {
            clsNum.push_back(validNum);
            cId = result[loop].classIdx;
            validNum = 1;
        }
    }
    clsNum.push_back(validNum);
    int idx = 0;
    int sumNum = 0;
    INFO_LOG("current class valid box number is: %d", clsNum[idx]);
    sumNum += clsNum[idx];
    size_t totalResultNum = result.size();
    for (size_t loop = 0; loop < totalResultNum; loop++) {
        if (loop == static_cast<size_t>(sumNum)) {
            idx++;
            INFO_LOG("current class valid box number is: %d", clsNum[idx]);
            sumNum += clsNum[idx];
        }
        INFO_LOG("class name: %s, class id: %d, confidence: %lf, lx: %lf, ly: %lf, rx: %lf, ry: %lf;",
            result[loop].className.c_str(),
            result[loop].classIdx,
            result[loop].confidence,
            result[loop].minX, result[loop].minY,
            result[loop].maxX, result[loop].maxY);
    }
}

string GetDetResultStr(const DetectBbox &detBbox)
{
    stringstream ss;
    ss << detBbox.className << "  ";
    ss << detBbox.classIdx << "  ";

    ss << fixed;

    auto floatStream = [](float val, int width, int precision, stringstream &ss) {
        ss << setw(width) << setprecision(precision) << val << "  ";
    };

    floatStream(detBbox.confidence, SCORE_INTERGER_WIDTH + SCORE_DECIMAL_WIDTH, SCORE_DECIMAL_WIDTH, ss);
    floatStream(detBbox.minX, COORD_INTERGER_WIDTH + COORD_DECIMAL_WIDTH, COORD_DECIMAL_WIDTH, ss);
    floatStream(detBbox.minY, COORD_INTERGER_WIDTH + COORD_DECIMAL_WIDTH, COORD_DECIMAL_WIDTH, ss);
    floatStream(detBbox.maxX, COORD_INTERGER_WIDTH + COORD_DECIMAL_WIDTH, COORD_DECIMAL_WIDTH, ss);
    floatStream(detBbox.maxY, COORD_INTERGER_WIDTH + COORD_DECIMAL_WIDTH, COORD_DECIMAL_WIDTH, ss);

    ss << endl;
    return ss.str();
}

void WriteResult(const vector<DetectBbox> &result, const vector<uint32_t> imageShape, const size_t classesNum)
{
    size_t resultSize = result.size();
    if (resultSize == 0) {
        INFO_LOG("The model does not detect any objects in the sample image, so no result.txt is generated.");
        return;
    }
    const string fileName = "result.txt";
    ofstream fout(fileName.c_str());

    stringstream ss;
    ss << imageShape[1] << "  " << imageShape[0] << "  " << classesNum << endl;
    fout << ss.str();
    ss.str("");
    if (fout.good() == false) {
        ERROR_LOG("fout open fail");
        return;
    }
    for (size_t loop = 0; loop < resultSize; loop++) {
        DetectBbox detectBbox = result[loop];
        if (detectBbox.confidence > 1.0f || detectBbox.confidence < 0.0f) {
            WARN_LOG("invalid score %f", detectBbox.confidence);
            continue;
        }
        ss << GetDetResultStr(detectBbox);
        fout << ss.str();
        ss.str("");
    }
    fout.close();
    return;
}
}

DetectPostprocess::DetectPostprocess(const DetectModelDesc &modelDesc)
{
    m_modelDesc = modelDesc;
    for (uint32_t k = 0; k < modelDesc.featmapStrides.size(); k++) {
        m_gridNumHeight.push_back((uint32_t)ceil((float)modelDesc.inputShape[0] / modelDesc.featmapStrides[k]));
        m_gridNumWidth.push_back((uint32_t)ceil((float)modelDesc.inputShape[1] / modelDesc.featmapStrides[k]));
    }
}

DetectPostprocess::~DetectPostprocess() { }

vector<DetectBbox> DetectPostprocess::PostProcess(ForwardEngine *forwardEngine, vector<uint32_t> &imageShape)
{
    
    svp_acl_mdl_io_dims aclOutDims;
    svp_acl_mdl_get_output_dims(forwardEngine->m_modelDesc, 0, &aclOutDims);


    m_modelDesc.imageShape[0] = imageShape[0];  
    m_modelDesc.imageShape[1] = imageShape[1];

    if (aclOutDims.dims[1]*aclOutDims.dims[1] != (m_gridNumHeight[0] * m_gridNumWidth[0])) {
        reverse(m_modelDesc.featmapStrides.begin(), m_modelDesc.featmapStrides.end());
        reverse(m_gridNumHeight.begin(), m_gridNumHeight.end());
        reverse(m_gridNumWidth.begin(), m_gridNumWidth.end());
    }

    // std::cout<<"dims:"<<m_modelDesc.featmapStrides[0]<<","<<m_modelDesc.featmapStrides[1]<<std::endl;
    // std::cout<<"dims:"<<m_gridNumWidth[0]<<","<<m_gridNumWidth[1]<<std::endl;

    struct timeval beginTime;
    struct timeval endTime;
    gettimeofday(&beginTime, NULL);

    vector<DetectBbox> result;
    const size_t boxInfoIdx = 0;
    
    for (uint32_t i = 0; i < m_modelDesc.featmapStrides.size(); i++) {
        size_t stride = 0;
        float *detectResult = forwardEngine->GetOutput(boxInfoIdx+i, stride);

        CalcBboxInfo(detectResult, result, i, stride);
    }
    
    vector<DetectBbox> newResult;
    newResult = _NMS(result);

    // std::cout<<newResult.size()<<std::endl;
    gettimeofday(&endTime, NULL);
    Utils::InfoDuration(beginTime, endTime, "PostProcess");
    return newResult;
}

vector<DetectBbox> DetectPostprocess::_NMS(vector<DetectBbox>& result)
{
    sort(result.begin(), result.end(), BboxConfDescend);
    NMS(result, m_modelDesc.nmsIouThresHold);
    vector<DetectBbox> newResult;
    for (DetectBbox &detectBbox : result) {
        if (detectBbox.mask != 0) {
            continue;
        }
        newResult.push_back(detectBbox);
    }
    return newResult;
}

void DetectPostprocess::DumpResult(vector<DetectBbox> &result)
{
    PrintResult(result);
    WriteResult(result, m_modelDesc.imageShape, m_modelDesc.totalClassName.size());
}

float DetectPostprocess::GetBboxMaxClassConfidence(const float *boxValue, uint32_t num, uint32_t &maxIndex) const
{
    LOG_CHECK_RETURN(boxValue == nullptr, FAILURE, "GetBbox boxValue nullptr error!");
    float maxExp = -FLT_MAX;

    for (uint32_t i = 0; i < num; ++i) {
        if (maxExp < boxValue[i]) {
            maxExp = boxValue[i];
            maxIndex = i;
        }
    }

    return maxExp;
}

map<string, float> DetectPostprocess::GetBboxSimilarClassConfidence(float classScore ,float objectScore, const float *boxValue, uint32_t num, uint32_t maxIndex)
{
    map<string, float> similarCl;
    float Score = 0;
    for (uint32_t i = 0; i < num; ++i) {
        if(maxIndex == i) continue;
        Score = Utils::Sigmoid(boxValue[i]);

        if((Score/classScore)>0.9)
        {
            similarCl[m_modelDesc.totalClassName[i]] = Score*objectScore;
        }
    }

    return similarCl;
}

bool DetectPostprocess::IsBboxConfInRange(uint32_t classIdx, float conf) const
{
    LOG_CHECK_RETURN(classIdx > m_modelDesc.classes.size() - 1, FAILURE, "ClassIdx large than class num err!");

    if (conf < m_modelDesc.classConfThres[classIdx]) {
        return false;
    }

    return true;
}

int32_t DetectPostprocess::CalcBboxInfo(const float *detectResult, vector<DetectBbox> &result,
    uint32_t outputIndex, size_t strideElemNum) const
{
   uint32_t maxValueIndex = 0;

    LOG_CHECK_RETURN(detectResult == nullptr, FAILURE, "Detection nullptr error!");
    vector<uint32_t> multiplierNums { m_gridNumHeight[outputIndex],
        m_gridNumWidth[outputIndex] };
    LOG_CHECK_RETURN(Utils::CheckU32MulUpperLimit(multiplierNums) == FAILURE, FAILURE,
        "Detection multiplier overflow error!");
    uint32_t gridNum = m_gridNumHeight[outputIndex] * m_gridNumWidth[outputIndex];
    for (uint32_t n = 0; n < gridNum; n++) {
        uint32_t w = n % m_gridNumWidth[outputIndex]; // Grid
        uint32_t h = n / m_gridNumWidth[outputIndex];
        uint32_t index = n * strideElemNum;
        float score = GetBboxMaxClassConfidence(&detectResult[index + OUTPUT_CLASS_INDEX],
            m_modelDesc.classes.size(), maxValueIndex);

        bool isBboxValid = IsBboxConfInRange(maxValueIndex, score);
        if (isBboxValid) {
            // printf("bb:%d,%d,%d\n",w,h,m_gridNumWidth[outputIndex]);
            // printf("aa:%f,%f,%f,%f\n",detectResult[index +0],detectResult[index +1],detectResult[index +2],detectResult[index +3]);

            DetectBbox detectBbox;
            detectBbox.minX = (float)((HALF + w - detectResult[index +
                OUTPUT_X1_INDEX]) * m_modelDesc.featmapStrides[outputIndex]);
            detectBbox.minY = (float)((HALF + h - detectResult[index +
                OUTPUT_Y1_INDEX]) * m_modelDesc.featmapStrides[outputIndex]);
            detectBbox.maxX = (float)((HALF + w + detectResult[index +
                OUTPUT_X2_INDEX]) * m_modelDesc.featmapStrides[outputIndex]);
            detectBbox.maxY = (float)((HALF + h + detectResult[index +
                OUTPUT_Y2_INDEX]) * m_modelDesc.featmapStrides[outputIndex]);
            detectBbox.confidence = score;
            detectBbox.classIdx = maxValueIndex;
            detectBbox.className = m_modelDesc.totalClassName[maxValueIndex];
            detectBbox.mask = 0;

            detectBbox.minX = Utils::Max(detectBbox.minX /
                m_modelDesc.inputShape[1] * m_modelDesc.imageShape[1], (float)0);
            detectBbox.minY = Utils::Max(detectBbox.minY /
                m_modelDesc.inputShape[0] * m_modelDesc.imageShape[0], (float)0);
            detectBbox.maxX = Utils::Min(detectBbox.maxX /
                m_modelDesc.inputShape[1] * m_modelDesc.imageShape[1], (float)m_modelDesc.imageShape[1]);
            detectBbox.maxY = Utils::Min(detectBbox.maxY /
                m_modelDesc.inputShape[0] * m_modelDesc.imageShape[0], (float)m_modelDesc.imageShape[0]);
            result.push_back(detectBbox);
        }
    }
    return SUCCESS;
}

float DetectPostprocess::CalcIou(const DetectBbox &detectBbox1, const DetectBbox &detectBbox2) const
{
    float interWidth = Utils::Min(detectBbox1.maxX, detectBbox2.maxX) - Utils::Max(detectBbox1.minX, detectBbox2.minX);
    float interHeight = Utils::Min(detectBbox1.maxY, detectBbox2.maxY) - Utils::Max(detectBbox1.minY, detectBbox2.minY);
    if (interWidth <= 0 || interHeight <= 0) {
        return 0;
    }
    float interArea = interWidth * interHeight;
    float box1Area = (detectBbox1.maxX - detectBbox1.minX) * (detectBbox1.maxY - detectBbox1.minY);
    float box2Area = (detectBbox2.maxX - detectBbox2.minX) * (detectBbox2.maxY - detectBbox2.minY);
    float unionArea = box1Area + box2Area - interArea;
    if (unionArea <= 0) {
        ERROR_LOG("detectBbox unionArea value error!");
        return 0;
    }
    return (interArea / unionArea);
}

void DetectPostprocess::NMS(vector<DetectBbox> &result, float nmsThresh) const
{
    vector<DetectBbox>::iterator compIter = result.begin();
    for (; compIter != result.end(); ++compIter) {
        DetectBbox &DetectBbox1 = *compIter;
        if (DetectBbox1.mask != 0) {
            continue;
        }
        vector<DetectBbox>::iterator compNextIter = compIter + 1;
        for (; compNextIter != result.end(); ++compNextIter) {
            DetectBbox &DetectBbox2 = *compNextIter;
            if (DetectBbox2.mask != 0) {
                continue;
            }
            float iou = CalcIou(DetectBbox1, DetectBbox2);
            if (iou > nmsThresh) {
                DetectBbox2.mask = 1;
            }
        }
    }
}