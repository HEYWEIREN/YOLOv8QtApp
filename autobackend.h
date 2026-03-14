#pragma once

#include <filesystem>             // 处理文件路径
#include <vector>                 // 向量容器
#include <unordered_map>         // 键值对容器，用于存储类别名
#include <opencv2/core/mat.hpp>  // OpenCV 图像类型
#include <opencv2/core/types.hpp> //  为 cv::RotatedRect 提供定义

#include "onnx_model_base.h"     // 基类：提供 ONNX 推理接口
#include "constants.h"           // 各种字符串常量（"stride"、"task" 等）

/**
 * @brief 表示单个目标的推理结果结构体
 */
struct YoloResults {
    int class_idx{};                   // 类别索引（0~79）
    float conf{};                      // 置信度（0~1）
    cv::Rect_<float> bbox;            // 检测框坐标
    cv::Mat mask;                     // 分割掩膜（仅 segment 模型）
    std::vector<float> keypoints{};    // 关键点（仅 pose 模型，格式 [x,y,c]*17）
    cv::RotatedRect rbox;              // ✅ OBB 旋转框
    bool has_rbox = false;             // ✅ 是否有效旋转框
};

/**
 * @brief 原图尺寸信息（用于后处理时坐标映射）
 */
struct ImageInfo {
    cv::Size raw_size;  // 原始图像尺寸
};

/**
 * @brief YOLOv8 模型的自动推理类（支持 detect、segment、pose）
 */
class AutoBackendOnnx : public OnnxModelBase {
public:
    // 构造函数：直接传入图片尺寸、stride、类别数等参数（适用于手动加载模型）
    AutoBackendOnnx(const char* modelPath, const char* logid, const char* provider,
                    const std::vector<int>& imgsz, const int& stride,
                    const int& nc, std::unordered_map<int, std::string> names);

    // 构造函数：从 ONNX 元数据中自动提取模型信息
    AutoBackendOnnx(const char* modelPath, const char* logid, const char* provider);

    // 获取模型输入图像尺寸
    virtual const std::vector<int>& getImgsz();
    virtual const int& getStride();              // 获取 stride 值
    virtual const int& getCh();                  // 获取输入通道数（通常为 3）
    virtual const int& getNc();                  // 获取类别数
    virtual const std::unordered_map<int, std::string>& getNames();  // 获取类别名 map
    virtual const std::vector<int64_t>& getInputTensorShape();       // 获取输入张量形状
    virtual const int& getWidth();               // 宽度
    virtual const int& getHeight();              // 高度
    virtual const cv::Size& getCvSize();         // OpenCV 尺寸（cv::Size）
    virtual const std::string& getTask();        // 返回模型任务类型（"detect"/"segment"/"pose"）

    /**
     * @brief 执行推理（重载3个接口，支持 Mat 和路径）
     * @param image 输入图像
     * @param conf 置信度阈值
     * @param iou IoU 阈值
     * @param mask_threshold 掩膜二值化阈值
     * @param conversionCode 图像颜色转换（如 cv::COLOR_BGR2RGB）
     * @return 推理结果数组
     */
    virtual std::vector<YoloResults> predict_once(cv::Mat& image, float& conf, float& iou, float& mask_threshold, int conversionCode = -1, bool verbose = true);
    virtual std::vector<YoloResults> predict_once(const std::filesystem::path& imagePath, float& conf, float& iou, float& mask_threshold, int conversionCode = -1, bool verbose = true);
    virtual std::vector<YoloResults> predict_once(const std::string& imagePath, float& conf, float& iou, float& mask_threshold, int conversionCode = -1, bool verbose = true);

    /**
     * @brief 将 OpenCV 图像填充成 blob（用于 ONNX 输入）
     */
    virtual void fill_blob(cv::Mat& image, float*& blob, std::vector<int64_t>& inputTensorShape);

    /**
     * @brief 后处理：分割任务（生成掩膜）
     */
    virtual void postprocess_masks(cv::Mat& output0, cv::Mat& output1, ImageInfo para, std::vector<YoloResults>& output,
                                   int& class_names_num, float& conf_threshold, float& iou_threshold,
                                   int& iw, int& ih, int& mw, int& mh, int& masks_features_num, float mask_threshold = 0.50f);

    /**
     * @brief 后处理：检测任务（生成框）
     */
    virtual void postprocess_detects(cv::Mat& output0, ImageInfo image_info, std::vector<YoloResults>& output,
                                     int& class_names_num, float& conf_threshold, float& iou_threshold);

    /**
     * @brief 后处理：姿态估计任务（关键点 + 框）
     */
    virtual void postprocess_kpts(cv::Mat& output0, ImageInfo& image_info, std::vector<YoloResults>& output,
                                  int& class_names_num, float& conf_threshold, float& iou_threshold);

    // 后处理：分类任务（输出 top1）
    virtual void postprocess_classify(cv::Mat& output0, std::vector<YoloResults>& output);

    virtual void postprocess_obb(cv::Mat& output0, ImageInfo image_info, std::vector<YoloResults>& output,
                                 int& class_names_num, float& conf_threshold, float& iou_threshold);


    /**
     * @brief 生成掩膜图像（内部工具函数）
     */
    static void _get_mask2(const cv::Mat& mask_info, const cv::Mat& mask_data, const ImageInfo& image_info, cv::Rect bound, cv::Mat& mask_out,
                           float& mask_thresh, int& iw, int& ih, int& mw, int& mh, int& masks_features_num, bool round_downsampled = false);


protected:
    std::vector<int> imgsz_;                             // 模型输入尺寸
    int stride_ = OnnxInitializers::UNINITIALIZED_STRIDE;  // 下采样比例（元数据中读取）
    int nc_ = OnnxInitializers::UNINITIALIZED_NC;        // 类别数
    int ch_ = 3;                                          // 通道数（默认3）
    std::unordered_map<int, std::string> names_;         // 类别名映射表
    std::vector<int64_t> inputTensorShape_;              // ONNX 输入张量 shape（如 [1,3,640,640]）
    cv::Size cvSize_;                                    // OpenCV 输入图像尺寸
    std::string task_;                                   // 当前模型任务类型：detect/segment/pose
};

