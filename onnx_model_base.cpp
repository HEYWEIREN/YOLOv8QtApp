#include "onnx_model_base.h"
#include <cpu_provider_factory.h>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

#include "constants.h"
#include "common.h"


/**
 * @brief Base class for any onnx model regarding the target.
 *
 * Wraps OrtApi.
 *
 * The caller provides a model path, logid, and provider.
 *
 * See the output logs for more information on warnings/errors that occur while processing the model.
 *
 * @param[in] modelPath Path to the model file.
 * @param[in] logid Log identifier.
 * @param[in] provider Provider (e.g., "CPU" or "CUDA"). (NOTE: for now only CPU is supported)
 */

OnnxModelBase::OnnxModelBase(const char* modelPath, const char* logid, const char* provider)
//: modelPath_(modelPath), env(std::move(env)), session(std::move(session))
    : modelPath_(modelPath)
{

     std::cout << "[DEBUG] provider string received: " << provider << std::endl;

    // TODO: too bad passing `ORT_LOGGING_LEVEL_WARNING` by default - for some cases
    //       info level would make sense too
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, logid);
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();

    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
    OrtCUDAProviderOptions cudaOption;

    std::string providerStr(provider);

    if (providerStr == OnnxProviders::CUDA) {
        if (cudaAvailable == availableProviders.end()) {
            std::cout << "CUDA is not supported by your ONNXRuntime build. Fallback to CPU." << std::endl;
        } else {
            sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
        }
    }
    else if (providerStr == OnnxProviders::CPU) {
        OrtSessionOptionsAppendExecutionProvider_CPU(sessionOptions, 0);
    }
    else {
        throw std::runtime_error("NotImplemented provider=" + providerStr);
    }

    std::cout << "Inference device: " << std::string(provider) << std::endl;
    #ifdef _WIN32
        auto modelPathW = get_win_path(modelPath);  // For Windows (wstring)
        session = Ort::Session(env, modelPathW.c_str(), sessionOptions);
    #else
        session = Ort::Session(env, modelPath, sessionOptions);  // For Linux (string)
    #endif
    //session = Ort::Session(env)
    // https://github.com/microsoft/onnxruntime/issues/14157
    //std::vector<const char*> inputNodeNames; //
    // ----------------
    // init input names
    inputNodeNames;
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings; // <-- newly added
    Ort::AllocatorWithDefaultOptions allocator;
    auto inputNodesNum = session.GetInputCount();
    for (int i = 0; i < inputNodesNum; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        inputNodeNameAllocatedStrings.push_back(std::move(input_name));
        inputNodeNames.push_back(inputNodeNameAllocatedStrings.back().get());
    }
    // -----------------
    // init output names
    outputNodeNames;
    auto outputNodesNum = session.GetOutputCount();
    std::vector<Ort::AllocatedStringPtr> outputNodeNameAllocatedStrings; // <-- newly added
    Ort::AllocatorWithDefaultOptions output_names_allocator;
    for (int i = 0; i < outputNodesNum; i++)
    {
        auto output_name = session.GetOutputNameAllocated(i, output_names_allocator);
        outputNodeNameAllocatedStrings.push_back(std::move(output_name));
        outputNodeNames.push_back(outputNodeNameAllocatedStrings.back().get());
    }
    // -------------------------
    // initialize model metadata
    model_metadata = session.GetModelMetadata();
    Ort::AllocatorWithDefaultOptions metadata_allocator;

    std::vector<Ort::AllocatedStringPtr> metadataAllocatedKeys = model_metadata.GetCustomMetadataMapKeysAllocated(metadata_allocator);
    std::vector<std::string> metadata_keys;
    metadata_keys.reserve(metadataAllocatedKeys.size());

    for (const Ort::AllocatedStringPtr& allocatedString : metadataAllocatedKeys) {
        metadata_keys.emplace_back(allocatedString.get());
    }

    // -------------------------
    // initialize metadata as the dict
    // even though we know exactly what metadata we intend to use
    // base onnx class should not have any ultralytics yolo-specific attributes like stride, task etc, so keep it clean as much as possible
    for (const std::string& key : metadata_keys) {
        Ort::AllocatedStringPtr metadata_value = model_metadata.LookupCustomMetadataMapAllocated(key.c_str(), metadata_allocator);
        if (metadata_value != nullptr) {
            auto raw_metadata_value = metadata_value.get();
            metadata[key] = std::string(raw_metadata_value);
        }
    }

    // initialize cstr
    for (const std::string& name : outputNodeNames) {
        outputNamesCStr.push_back(name.c_str());
    }

    for (const std::string& name : inputNodeNames)
    {
        inputNamesCStr.push_back(name.c_str());
    }

    auto dump_tensor_io = [&](bool is_input) {
        size_t n = is_input ? session.GetInputCount() : session.GetOutputCount();
        for (size_t i = 0; i < n; ++i) {
            auto typeInfo = is_input ? session.GetInputTypeInfo(i) : session.GetOutputTypeInfo(i);
            auto onnxType = typeInfo.GetONNXType();
            std::cout << "[IO] " << (is_input ? "Input" : "Output") << "[" << i << "] "
                      << (is_input ? inputNodeNames[i] : outputNodeNames[i]) << " ";

            if (onnxType == ONNX_TYPE_TENSOR) {
                auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
                auto shape = tensorInfo.GetShape();
                auto elemType = tensorInfo.GetElementType();
                std::cout << "elemType=" << (int)elemType << " shape=[";
                for (size_t k = 0; k < shape.size(); ++k) {
                    std::cout << shape[k] << (k + 1 == shape.size() ? "" : ",");
                }
                std::cout << "]";
            } else {
                std::cout << "onnxType=" << (int)onnxType; // 避免非tensor崩
            }
            std::cout << std::endl;
        }
    };

    std::cout << "[IO] model=" << modelPath_ << std::endl;
    dump_tensor_io(true);
    dump_tensor_io(false);


}

const std::vector<std::string>& OnnxModelBase::getInputNames() {
    return inputNodeNames;
}

const std::vector<std::string>& OnnxModelBase::getOutputNames() {
    return outputNodeNames;
}

const Ort::ModelMetadata& OnnxModelBase::getModelMetadata()
{
    return model_metadata;
}

const std::unordered_map<std::string, std::string>& OnnxModelBase::getMetadata()
{
    return metadata;
}


const Ort::Session& OnnxModelBase::getSession()
{
    return session;
}

const char* OnnxModelBase::getModelPath()
{
    return modelPath_;
}

const std::vector<const char*> OnnxModelBase::getOutputNamesCStr()
{
    return outputNamesCStr;
}

const std::vector<const char*> OnnxModelBase::getInputNamesCStr()
{
    return inputNamesCStr;
}

std::vector<Ort::Value> OnnxModelBase::forward(std::vector<Ort::Value>& inputTensors)
{
    // todo: make runOptions parameter here

    return session.Run(Ort::RunOptions{ nullptr },
        inputNamesCStr.data(),
        inputTensors.data(),
        inputNamesCStr.size(),
        outputNamesCStr.data(),
        outputNamesCStr.size());
}

//OnnxModelBase::~OnnxModelBase() {
//    // empty body
//}
