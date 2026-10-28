#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/face.hpp>
#include <string>
#include <vector>

namespace TA::FaceLabel {
    class Inference {
    public:
        struct FaceResult {
            cv::Rect box;
            float score = 0.f;
            cv::Point2f landmarks[5];
        };

        explicit Inference(const std::string& model_path);
        Inference() = default;
        ~Inference() = default;

        std::vector<FaceResult> Run(const cv::Mat& frame, int min_face_size = 0);

    private:
        cv::Ptr<cv::FaceDetectorYN> detector_;
        void InitModel(const std::string& model_path);
    };
}