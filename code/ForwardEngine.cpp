#include <iomanip>
#include <map>
#include <string>
#include <sstream>
#include <algorithm>
#include <fstream>
#include "ForwardEngine.h"

ForwardEngine::ForwardEngine(std::string &modelPath)
{
    m_modelPath = modelPath;
}

ForwardEngine::~ForwardEngine()
{
    DestroyInput();
    DestroyOutput();
    UnloadModel();
    DestroyBaseResource();
}

int32_t ForwardEngine::PrepareResource(void *inputDataBuf)
{
    LOG_CHECK_RETURN(CreateBaseResource() != SVP_ACL_SUCCESS, FAILURE, "sample init resource failed");
    LOG_CHECK_RETURN(LoadModel() != SVP_ACL_SUCCESS, FAILURE, "execute LoadModel failed");
    LOG_CHECK_RETURN(CreateOutput() != SVP_ACL_SUCCESS, FAILURE, "execute CreateOutput failed");
    LOG_CHECK_RETURN(CreateInput(inputDataBuf) != SVP_ACL_SUCCESS, FAILURE, "CreateInputBuf failed");
    if (inputDataBuf != nullptr) {
        m_input0BufFromUser = true;
    }
    return SUCCESS;
}

int32_t ForwardEngine::Forward()
{
    struct timeval beginTime;
    struct timeval endTime;
    gettimeofday(&beginTime, nullptr);
    LOG_CHECK_RETURN(svp_acl_mdl_execute(m_modelId, m_input, m_output) != SVP_ACL_SUCCESS, FAILURE,
        "execute model failed, modelId is %u", m_modelId);
    gettimeofday(&endTime, nullptr);
    Utils::InfoDuration(beginTime, endTime, "Forward");
    return SUCCESS;
}

float *ForwardEngine::GetOutput(size_t outputIdx, size_t &stride)
{
    stride = svp_acl_mdl_get_output_default_stride(m_modelDesc, outputIdx) / sizeof(float);
    svp_acl_data_buffer *dataBuffer = svp_acl_mdl_get_dataset_buffer(m_output, outputIdx);
    void *bufferAddr = svp_acl_get_data_buffer_addr(dataBuffer);
    auto *output = static_cast<float *>(bufferAddr);
    return output;
}

// this step includs creating device, context and stream
int32_t ForwardEngine::CreateBaseResource()
{
    LOG_CHECK_RETURN(svp_acl_init(nullptr) != SVP_ACL_SUCCESS, FAILURE, "acl init failed");
    LOG_CHECK_RETURN(svp_acl_rt_set_device(m_deviceId) != SVP_ACL_SUCCESS, FAILURE, "acl open device %d failed",
        m_deviceId);
    LOG_CHECK_RETURN(svp_acl_rt_set_op_wait_timeout(0) != SVP_ACL_SUCCESS, FAILURE, "acl set op wait time failed");
    LOG_CHECK_RETURN(svp_acl_rt_create_context(&m_context, m_deviceId) != SVP_ACL_SUCCESS, FAILURE,
        "acl create context failed");
    LOG_CHECK_RETURN(svp_acl_rt_create_stream(&m_stream) != SVP_ACL_SUCCESS, FAILURE, "acl create stream failed");

    svp_acl_rt_run_mode runMode;
    LOG_CHECK_RETURN(svp_acl_rt_get_run_mode(&runMode) != SVP_ACL_SUCCESS, FAILURE, "acl get run mode failed");
    LOG_CHECK_RETURN(runMode != SVP_ACL_DEVICE, FAILURE, "acl run mode failed");
    return SUCCESS;
}

int32_t ForwardEngine::LoadModel()
{
    uint32_t fileSize = 0;
    m_modelMem = Utils::ReadModelFile(m_modelPath, fileSize);
    svp_acl_error ret = svp_acl_mdl_load_from_mem(static_cast<uint8_t *>(m_modelMem), fileSize, &m_modelId);
    LOG_CHECK_RETURN(ret != SVP_ACL_SUCCESS, FAILURE, "svp_acl_mdl_load_from_mem failed");

    m_modelDesc = svp_acl_mdl_create_desc();
    LOG_CHECK_RETURN(m_modelDesc == nullptr, FAILURE, "svp_acl_mdl_create_desc failed");
    ret = svp_acl_mdl_get_desc(m_modelDesc, m_modelId);
    LOG_CHECK_RETURN(ret != SVP_ACL_SUCCESS, FAILURE, "svp_acl_mdl_get_desc failed");
    m_loadFlag = true;
    return SUCCESS;
}


int32_t ForwardEngine::CreateInput(void *inputDataBuf)
{
    m_input = svp_acl_mdl_create_dataset();
    LOG_CHECK_RETURN(m_input == nullptr, FAILURE, "CreateInput svp_acl_mdl_create_dataset failed");
    size_t numInputs = svp_acl_mdl_get_num_inputs(m_modelDesc);
    std::cout<<numInputs<<std::endl;
    for (size_t i = 0; i < numInputs; ++i) {
        size_t stride = svp_acl_mdl_get_input_default_stride(m_modelDesc, i);
        size_t bufSize = svp_acl_mdl_get_input_size_by_index(m_modelDesc, i);
        if (i == 0 && inputDataBuf != nullptr) {
            LOG_CHECK_RETURN(CreateDatasetBuf(m_input, bufSize, stride, inputDataBuf) != SUCCESS, FAILURE,
                "CreateDatasetBuf failed");
        } else {
            LOG_CHECK_RETURN(CreateDatasetBuf(m_input, bufSize, stride, nullptr) != SUCCESS, FAILURE,
                "CreateDatasetBuf failed");
        }
    }
    return SUCCESS;
}


int32_t ForwardEngine::CreateOutput()
{
    m_output = svp_acl_mdl_create_dataset();
    LOG_CHECK_RETURN(m_output == nullptr, FAILURE, "CreateOutput svp_acl_mdl_create_dataset failed");
    size_t numOutputs = svp_acl_mdl_get_num_outputs(m_modelDesc);
    
    for (size_t i = 0; i < numOutputs; ++i) {
        size_t stride = svp_acl_mdl_get_output_default_stride(m_modelDesc, i);
        
        size_t bufSize = svp_acl_mdl_get_output_size_by_index(m_modelDesc, i);
        
        LOG_CHECK_RETURN(CreateDatasetBuf(m_output, bufSize, stride, nullptr) != SUCCESS, FAILURE,
            "CreateDatasetBuf failed");
    }

    return SUCCESS;
}

int32_t ForwardEngine::CreateDatasetBuf(svp_acl_mdl_dataset *dataset, size_t &bufSize, size_t &stride,
    void *allocatedBuf)
{
    void *bufPtr = allocatedBuf;
    if (bufPtr == nullptr) {
        svp_acl_error ret = svp_acl_rt_malloc(&bufPtr, bufSize, SVP_ACL_MEM_MALLOC_NORMAL_ONLY);
        LOG_CHECK_RETURN(ret != SVP_ACL_SUCCESS, FAILURE, "svp_acl_rt_malloc failed. malloc size is %zu", bufSize);
        Utils::InitData(static_cast<int8_t *>(bufPtr), bufSize);
    }

    svp_acl_data_buffer *dataBuffer = svp_acl_create_data_buffer(bufPtr, bufSize, stride);
    if (dataBuffer == nullptr) {
        ERROR_LOG("svp_acl_create_data_buffer failed");
        svp_acl_rt_free(bufPtr);
        bufPtr = nullptr;
        return FAILURE;
    }

    svp_acl_error ret = svp_acl_mdl_add_dataset_buffer(dataset, dataBuffer);
    if (ret != SVP_ACL_SUCCESS) {
        ERROR_LOG("svp_acl_mdl_add_dataset_buffer failed, and ret: %x", ret);
        svp_acl_rt_free(bufPtr);
        svp_acl_destroy_data_buffer(dataBuffer);
        bufPtr = nullptr;
        dataBuffer = nullptr;
        return FAILURE;
    }

    return SUCCESS;
}


void ForwardEngine::UnloadModel()
{
    if (!m_loadFlag) {
        WARN_LOG("no model had been loaded, unload failed");
        return;
    }

    if (m_modelDesc != nullptr) {
        (void)svp_acl_mdl_destroy_desc(m_modelDesc);
        m_modelDesc = nullptr;
    }

    (void)svp_acl_mdl_unload(m_modelId);

    if (m_modelMem != nullptr) {
        svp_acl_rt_free(m_modelMem);
        m_modelMem = nullptr;
    }

    m_loadFlag = false;
}


void ForwardEngine::DestroyInput()
{
    DestroyDatasetBuf(m_input);
    m_input = nullptr;
}

void ForwardEngine::DestroyOutput()
{
    DestroyDatasetBuf(m_output);
    m_output = nullptr;
}

void ForwardEngine::DestroyDatasetBuf(svp_acl_mdl_dataset *dataset)
{
    if (dataset == nullptr) {
        return;
    }

    for (size_t i = 0; i < svp_acl_mdl_get_dataset_num_buffers(dataset); ++i) {
        svp_acl_data_buffer *dataBuffer = svp_acl_mdl_get_dataset_buffer(dataset, i);
        void *tmp = svp_acl_get_data_buffer_addr(dataBuffer);
        if (i != 0 || !m_input0BufFromUser) {
            svp_acl_rt_free(tmp);
        }
        svp_acl_destroy_data_buffer(dataBuffer);
    }
    svp_acl_mdl_destroy_dataset(dataset);
}

void ForwardEngine::DestroyBaseResource()
{
    if (m_stream != nullptr) {
        (void)svp_acl_rt_destroy_stream(m_stream);
        m_stream = nullptr;
    }
    if (m_context != nullptr) {
        (void)svp_acl_rt_destroy_context(m_context);
        m_context = nullptr;
    }
    (void)svp_acl_rt_reset_device(m_deviceId);
    (void)svp_acl_finalize();
}
