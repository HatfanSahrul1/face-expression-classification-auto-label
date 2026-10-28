#include <vector>
#include "face_label/FaceLabel.h"
#include <iostream>

#include <iostream>

int main() {
    TA::FaceLabel::FaceLabel labeler(
        "model/face_detection_yunet_2026may.onnx", 
        "/home/hatfan/models/expression/model.onnx"
    );

    std::vector<TA::FaceLabel::FaceLabel::VideoConfig> videos = {
        {"/home/hatfan/dataset/raw/6.mp4", 1},
    };

    labeler.ProcessVideos(videos, "dataset_emotion");

    std::cout << "Selesai membuat dataset!\n";
    return 0;
}