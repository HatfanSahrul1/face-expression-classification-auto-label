#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <fstream>
#include "inference/inference.h"
#include "face_extractor/face_extractor.h"

namespace TA::FaceLabel {

    class FaceLabel {
    private:
        Inference face_detector_;
        cv::dnn::Net emotion_net_;
        
        const std::vector<std::string> emotions_ = {
            "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
        };

        cv::Mat PreprocessEmotion(const cv::Mat& face_img);
        
        void SaveDataset(const cv::Mat& face_img, const std::string& label, 
                         const std::string& video_name, double timestamp, float score, 
                         std::ofstream& csv_file, const std::string& out_dir);

    public:
        struct VideoConfig {
            std::string path;
            double interval_sec;
        };

        FaceLabel(const std::string& yunet_path, const std::string& emotion_onnx_path);
        
        void ProcessVideos(const std::vector<VideoConfig>& videos, const std::string& output_dataset_dir);
    };
}