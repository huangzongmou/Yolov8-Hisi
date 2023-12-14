#ifndef FORWARD_ENGINE_H
#define FORWARD_ENGINE_H

#include "Utils.h"

class ForwardEngine {
public:
    svp_acl_mdl_desc *m_modelDesc { nullptr };
    svp_acl_mdl_dataset *m_output { nullptr };

    explicit ForwardEngine(string &modelPath);
    ~ForwardEngine();
    int32_t PrepareResource(void *inputDataBuf);
    int32_t Forward();
    float *GetOutput(size_t outputIdx, size_t &stride);

private:
    bool m_input0BufFromUser {false};
    std::string m_modelPath;
    int32_t m_deviceId { 0 };
    svp_acl_rt_context m_context { nullptr };
    svp_acl_rt_stream m_stream { nullptr };
    uint32_t m_modelId { 0 };
    void *m_modelMem { nullptr };

    svp_acl_mdl_dataset *m_input { nullptr };
    bool m_loadFlag { false };

    int32_t CreateBaseResource();
    int32_t LoadModel();
    int32_t CreateInput(void *inputDataBuf);
    int32_t CreateOutput();
    int32_t CreateDatasetBuf(svp_acl_mdl_dataset *dataset, size_t &bufSize, size_t &stride, void *allocatedBuf);
    void UnloadModel();
    void DestroyInput();
    void DestroyOutput();
    void DestroyBaseResource();
    void DestroyDatasetBuf(svp_acl_mdl_dataset *dataset);
};
#endif // FORWARD_ENGINE_H