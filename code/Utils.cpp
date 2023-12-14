#include "Utils.h"

int32_t Utils::CheckU32MulUpperLimit(const std::vector<uint32_t> &numbers)
{
    if (numbers.size() <= 1) {
        return SUCCESS;
    }
    uint32_t multiplierTemp = numbers[0];
    for (uint32_t i = 1; i < numbers.size(); i++) {
        if ((numbers[i] != 0) && (multiplierTemp > UINT_MAX / numbers[i])) {
            ERROR_LOG("Multiplication over upper limit!");
            return FAILURE;
        } else {
            multiplierTemp *= numbers[i];
        }
    }
    return SUCCESS;
}

float Utils::FastExpFloat(float x)
{
    union {
        uint32_t i;
        float f;
    } v {};
    /* 23 1.4426950409  126.93490512f: fast exp coeff */
    v.i = static_cast<uint32_t>((1 << 23) * (1.4426950409 * x + 126.93490512f));
    return v.f;
}

float Utils::FastSigmoid(float x)
{
    return 1.0f / (1.0f + FastExpFloat(-x));
}

float Utils::Sigmoid(float value)
{
    return (1.0f / (1 + std::exp(-value)));
}

uint32_t Utils::Align2(uint32_t width)
{
    /* 2: align by 2 */
    const uint32_t alignSize = 2;
    const uint64_t alignMax = static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
    uint64_t extendWidth = static_cast<uint64_t>(width + alignSize - 1);
    LOG_CHECK_RETURN(extendWidth > alignMax, 0, "Align calculation is out of bounds!");
    uint32_t num = ((width + alignSize - 1) / alignSize * alignSize);
    return num;
}

bool Utils::FloatCompare(float num1, float num2)
{
    float diff = std::fabs(num1 - num2);
    return (diff <= FLT_EPSILON) ? true : false;
}

void Utils::InitData(int8_t *data, size_t dataSize)
{
    for (size_t i = 0; i < dataSize; i++) {
        data[i] = 0;
    }
}

void *Utils::ReadModelFile(const std::string &fileName, uint32_t &fileSize)
{
    std::ifstream binFile(fileName, std::ifstream::binary);
    LOG_CHECK_RETURN(!binFile.is_open(), nullptr, "open file %s failed", fileName.c_str());

    binFile.seekg(0, binFile.end);
    int binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        ERROR_LOG("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return nullptr;
    }
    binFile.seekg(0, binFile.beg);
    void *binFileBufferData = nullptr;
    svp_acl_error ret = svp_acl_rt_malloc(&binFileBufferData, binFileBufferLen, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("svp_acl_rt_malloc failed. malloc size is %u", binFileBufferLen);
        binFile.close();
        return nullptr;
    }

    InitData(static_cast<int8_t *>(binFileBufferData), binFileBufferLen);

    binFile.read(static_cast<char *>(binFileBufferData), binFileBufferLen);
    binFile.close();
    fileSize = static_cast<uint32_t>(binFileBufferLen);
    return binFileBufferData;
}

int32_t Utils::GetAlign(uint32_t width, const uint32_t align)
{
    if (align == 0) {
        return 0;
    }
    uint32_t num = ((width + align - 1) / align * align);
    return num;
}

void Utils::CreateImageBuf(uint32_t width, uint32_t height, ot_video_frame &targetImage, uint32_t align)
{
    uint32_t channel = 3;
    uint32_t half = 2;
    uint32_t stride = GetAlign(width, align);
    td_phys_addr_t phyAddr = 0;
    void *virAddr = nullptr;
    uint32_t fileSize = height * stride * channel / half; /* 3: get mmz required, 2: mmz required */
    int ret = ss_mpi_sys_mmz_alloc(&phyAddr, &virAddr, nullptr, nullptr, fileSize);
    LOG_CHECK_PRINTF(ret != SUCCESS, "ss_mpi_sys_mmz_alloc failed, ret=0x%x!", ret);
    targetImage.width = width;
    targetImage.height = height;
    targetImage.stride[0] = stride;
    targetImage.phys_addr[0] = phyAddr;
    targetImage.virt_addr[0] = (void *)(virAddr);
    targetImage.phys_addr[1] = targetImage.phys_addr[0] + targetImage.height * targetImage.stride[0];
    targetImage.virt_addr[1] =
        (void *)((td_phys_addr_t)targetImage.virt_addr[0] + targetImage.height * targetImage.stride[0]);
    targetImage.stride[1] = targetImage.stride[0];
}

void Utils::ReleaseImageBuf(ot_video_frame &targetImage)
{
    if (targetImage.virt_addr[0] != nullptr) {
        ss_mpi_sys_mmz_free(targetImage.phys_addr[0], targetImage.virt_addr[0]);
        targetImage.phys_addr[0] = 0;
        targetImage.virt_addr[0] = nullptr;
        targetImage.phys_addr[1] = 0;
        targetImage.virt_addr[1] = nullptr;
    }
}

void Utils::InfoDuration(timeval &beginTime, timeval &endTime, string phase)
{
    unsigned usTos = 1000000;
    unsigned processTime = usTos * (endTime.tv_sec - beginTime.tv_sec) + endTime.tv_usec - beginTime.tv_usec;
    INFO_LOG("%s Duration %.1fus", phase.c_str(), static_cast<double>(processTime));
}

int32_t Utils::ReadImageFile(const std::string &fileName, vector<uint32_t> &imageShape,
    vector<uint32_t> &modelInputShape, ot_video_frame &modelInput)
{
    struct timeval beginTime;
    struct timeval endTime;
    uint32_t channel = 3;
    uint32_t half = 2;
    gettimeofday(&beginTime, nullptr);
    CreateImageBuf(modelInputShape[1], modelInputShape[0], modelInput, ALIGN_16);
    constexpr size_t count = 1;
    // std::cout<<modelInputShape[1];
    FILE *fd = fopen(fileName.c_str(), "rb");

    if (imageShape == modelInputShape) {
        size_t countRead = fread((void *)modelInput.virt_addr[0],
            modelInputShape[0] * modelInputShape[1] * channel / half, count, fd); // *3/2: calculate yuv size
        gettimeofday(&endTime, nullptr);
        InfoDuration(beginTime, endTime, "PreProcess");
        if (countRead != count) {
            goto RELEASE_MODEL_INPUT;
        }
    } else {
        ot_video_frame tempImage = {};
        CreateImageBuf(imageShape[1], imageShape[0], tempImage, ALIGN_2);
        size_t countRead = fread((void *)tempImage.virt_addr[0], imageShape[0] * imageShape[1] * channel / half, count,
            fd); // *3/2: calculate yuv size
        if (countRead != count) {
            ReleaseImageBuf(tempImage);
            ReleaseImageBuf(modelInput);
            fclose(fd);
            ERROR_LOG("image open failed, please check imageShape in cfg file");
            return FAILURE;
        }
        ImageResize(tempImage, modelInput);
        gettimeofday(&endTime, nullptr);
        InfoDuration(beginTime, endTime, "PreProcess");
        ReleaseImageBuf(tempImage);
    }
    fclose(fd);
    return SUCCESS;

RELEASE_MODEL_INPUT:
    ReleaseImageBuf(modelInput);
    fclose(fd);
    ERROR_LOG("image open failed");
    return FAILURE;
}

static void ConvertToVideoFrame(const ot_video_frame &srcFrame, ot_video_frame_info &dstFrame)
{
    dstFrame.video_frame.compress_mode = OT_COMPRESS_MODE_NONE;
    dstFrame.video_frame.pixel_format = OT_PIXEL_FORMAT_YVU_SEMIPLANAR_420;
    dstFrame.video_frame.height = srcFrame.height;
    dstFrame.video_frame.width = srcFrame.width;
    dstFrame.video_frame.stride[0] = srcFrame.stride[0];
    dstFrame.video_frame.stride[1] = srcFrame.stride[0];
    dstFrame.video_frame.phys_addr[0] = srcFrame.phys_addr[0];
    dstFrame.video_frame.phys_addr[1] = srcFrame.phys_addr[1];
    dstFrame.video_frame.virt_addr[0] = srcFrame.virt_addr[0];
    dstFrame.video_frame.virt_addr[1] = srcFrame.virt_addr[1];
}

int32_t Utils::ImageResize(ot_video_frame &image, ot_video_frame &dstImage)
{
    ot_vgs_handle vgsHandle;
    int ret = ss_mpi_vgs_begin_job(&vgsHandle);
    LOG_CHECK_RETURN(ret != SUCCESS, ret, "VGS begin job failed with %#x!\n", ret);

    ot_vgs_task_attr taskAttr;
    (void)memset_s(&taskAttr, sizeof(taskAttr), 0, sizeof(taskAttr));
    ConvertToVideoFrame(image, taskAttr.img_in);
    ConvertToVideoFrame(dstImage, taskAttr.img_out);

    ot_vgs_scale_coef_mode scaleCoefMode = OT_VGS_SCALE_COEF_NORM;

    ret = ss_mpi_vgs_add_scale_task(vgsHandle, &taskAttr, scaleCoefMode);
    if (ret != SUCCESS) {
        ERROR_LOG("AddScaleTask failed with %#x!\n", ret);
        ss_mpi_vgs_cancel_job(vgsHandle);
        return ret;
    }
    ret = ss_mpi_vgs_end_job(vgsHandle);
    if (ret != SUCCESS) {
        ERROR_LOG("VGS end job failed with %#x!\n", ret);
        ss_mpi_vgs_cancel_job(vgsHandle);
        return ret;
    }
    return SUCCESS;
}

int32_t Utils::CheckFile(const std::string &fileName)
{
    struct stat sBuf;
    int fileStatus = 0;
    fileStatus = stat(fileName.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", fileName.c_str());
        return FAILURE;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", fileName.c_str());
        return FAILURE;
    }
    return SUCCESS;
}

vector<uint32_t> Utils::StringToInt(vector<string> result)
{
    vector<uint32_t> new_result;
    for (string s : result) {
        new_result.push_back(atoi(s.c_str()));
    }
    return new_result;
}

vector<float> Utils::StringToFloat(vector<string> result)
{
    vector<float> new_result;
    for (string s : result) {
        new_result.push_back(atof(s.c_str()));
    }
    return new_result;
}

string Utils::FilterString(string const & val, char const * target = " \t\r\n")
{
    string result(val);
    string::size_type index = result.find_last_not_of(target);
    if (index != string::npos) {
        result.erase(++index);
    }

    index = result.find_first_not_of(target);
    if (index != string::npos) {
        result.erase(0, index);
    } else {
        result.erase();
    }
    return result;
}

vector<string> Utils::SplitString(string str, string pattern)
{
    string::size_type pos;
    vector<string> result;
    str += pattern;
    uint32_t size = str.size();
    for (uint32_t i = 0; i < size; i++) {
        pos = str.find(pattern, i);
        if (pos < size) {
            string strFilterSpace = FilterString(str.substr(i, pos - i), " \t\r\n");
            string strFilterDict = FilterString(FilterString(strFilterSpace, "{"), "}");
            string s = FilterString(FilterString(strFilterDict, "["), "]");
            result.push_back(s);
            i = pos + pattern.size() - 1;
        }
    }
    return result;
}

bool Utils::File2Map(string &cfgPath, map<string, string> &cfgData)
{
    struct stat sBuf;
    int32_t fileStatus = stat(cfgPath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", cfgPath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", cfgPath.c_str());
        return false;
    }

    fstream f;
    f.open(cfgPath.c_str(), fstream::in);
    if (!f.is_open()) {
        return false;
    }
    string line;
    int32_t index = -1;
    while (getline(f, line)) {
        index++;
        // Skip Comments and empty lines
        if (!line.length()) {
            continue;
        }
        if (line[0] == '#') {
            continue;
        }
        if (line[0] == '/') {
            continue;
        }

        int32_t targetIndex = line.find('=');
        if (targetIndex == -1) {
            WARN_LOG("WARNING: Statement '%s' in file %s:%d is invalid and therefor will be ignored!", line.c_str(),
                cfgPath.c_str(), index);
            continue;
        }
        string key = FilterString(line.substr(0, targetIndex));
        string value = FilterString(line.substr(targetIndex + 1));

        if (cfgData[key] != "") {
            WARN_LOG("WARNING: Statement '%s' in file %s:%d redefines a value!", line.c_str(), cfgPath.c_str(), index);
        }
        cfgData[key] = value;
    }
    f.close();
    return true;
}

string Utils::GetValue(map<string, string> &cfgData, string key)
{
    string val = cfgData[key];
    if (val == "") {
        WARN_LOG("WARNING: cfg file was not defined in [%s]! Value is undefined!", key.c_str());
    }
    return val;
}
