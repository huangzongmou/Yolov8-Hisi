#ifndef DETECT_CFG_PARSE_H
#define DETECT_CFG_PARSE_H

#include "Utils.h"
#include "DetectPostprocess.h"

namespace cfg {
using namespace postprocess;
class DetectCfgParse {
public:
	static int32_t CfgParse(string &cfgPath, string &modelPath, string &imagePath, DetectModelDesc &modelDesc);
};
}
#endif // DETECT_CFG_PARSE_H
