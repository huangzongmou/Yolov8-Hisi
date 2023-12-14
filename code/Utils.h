#ifndef UTILS_H
#define UTILS_H
#undef INT_MAX
#define INT_MAX __INT_MAX__
#undef UINT_MAX
#define UINT_MAX (INT_MAX * 2U + 1U)
#define INFO_LOG(fmt, ...) fprintf(stdout, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define WARN_LOG(fmt, ...) fprintf(stdout, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define ERROR_LOG(fmt, ...) fprintf(stdout, "[ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOG_CHECK_PRINTF(val, fmt, ...)    \
    do {                                   \
        if ((val)) {                       \
            ERROR_LOG(fmt, ##__VA_ARGS__); \
        }                                  \
    } while (0)

#define LOG_CHECK_RETURN(val, ret, fmt, ...) \
    do {                                     \
        if ((val)) {                         \
            ERROR_LOG(fmt, ##__VA_ARGS__);   \
            return (ret);                    \
        }                                    \
    } while (0)

#include <fstream>
#include <sys/stat.h>
#include <sys/time.h>
#include <bits/stdc++.h>
#include "securec.h"
#include "svp_acl.h"
#include "svp_acl_mdl.h"
#include "ot_common_video.h"
#include "ss_mpi_sys.h"
#include "ss_mpi_vgs.h"

using namespace std;
const int32_t SUCCESS = 0;
const int32_t  FAILURE = -1;
const uint32_t ALIGN_16 = 16;
const uint32_t ALIGN_2 = 2;
const int BYTE_BIT_NUM = 8;

class Utils {
public:
    static int32_t CheckU32MulUpperLimit(const std::vector<uint32_t> &numbers);
    static float FastExpFloat(float x);
    static float FastSigmoid(float x);
    static float Sigmoid(float value);
    static uint32_t Align2(uint32_t width);
    static bool FloatCompare(float num1, float num2);
    
    static void InitData(int8_t* data, size_t dataSize);
    static int32_t GetAlign(uint32_t width, const uint32_t align);
    static void* ReadModelFile(const std::string &fileName, uint32_t &fileSize);
    static int32_t ReadImageFile(const std::string &fileName, vector<uint32_t> &imageShape, vector<uint32_t> &modelInputShape, ot_video_frame &modelInput);
    static void ConstructVideoFrame(ot_video_frame &targetImage, td_phys_addr_t phyAddr, void *virAddr, uint32_t width, uint32_t height);
    static void CreateImageBuf(uint32_t width, uint32_t height, ot_video_frame &targetImage, uint32_t align);
    static void ReleaseImageBuf(ot_video_frame &targetImage);
    static int32_t ImageResize(ot_video_frame &image, ot_video_frame &dstImage);
    static int32_t CheckFile(const std::string& fileName);
    static void InfoDuration(timeval &begin_time, timeval &end_time, string phase);

    static vector<uint32_t> StringToInt(vector<string> result);
    static vector<float> StringToFloat(vector<string> result);
    static vector<string> SplitString(string str, string pattern);
    static string FilterString(string const &val, char const *target);
    static bool File2Map(string &cfgPath, map<string, string> &cfgData);
    static string GetValue(map<string, string> &cfgData, string key);

    template <typename T> static T Max(const T &a, const T &b)
    {
        return (a > b ? a : b);
    }
    template <typename T> static T Min(const T &a, const T &b)
    {
        return (a < b ? a : b);
    }
};
#endif
