#include "face_extractor.h"

namespace TA::FaceLabel {
    cv::Mat FaceExtractor::CropFace(const cv::Mat& frame, const Inference::FaceResult& face) {
        if (face.box.width <= 0 || face.box.height <= 0) return cv::Mat();
        
        cv::Mat cropped = frame(face.box).clone();
        
        cv::Mat resized;
        cv::resize(cropped, resized, cv::Size(128, 128), 0, 0, cv::INTER_CUBIC);
        
        return resized;
    }
}