#include <iostream>
#include <vector>
#include <string>
#include <array>
#include <sstream>
#include <onnxruntime_cxx_api.h>

// Gesture label names — must match training order
const std::vector<std::string> GESTURE_NAMES = {
    "open_hand", "fist", "point_up", "peace", "thumbs_up"
};

// Takes 63 landmark floats, runs inference, returns gesture name
std::string predict_gesture(
    Ort::Session& session,
    Ort::AllocatorWithDefaultOptions& allocator,
    const std::vector<float>& landmarks)
{
    // Input shape: [1, 63]
    std::array<int64_t, 2> input_shape = {1, 63};

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float*>(landmarks.data()),
        landmarks.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Run inference
    const char* input_names[]  = {"landmarks"};
    const char* output_names[] = {"gesture"};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    // Get output scores
    float* scores = output_tensors[0].GetTensorMutableData<float>();

    // Find highest score (argmax)
    int best = 0;
    for (int i = 1; i < (int)GESTURE_NAMES.size(); i++) {
        if (scores[i] > scores[best]) best = i;
    }

    return GESTURE_NAMES[best];
}

int main()
{
    // Path to your ONNX model
    // const wchar_t* model_path = L"..\\..\\models\\gesture_model.onnx";
    const wchar_t* model_path = L"C:\\Users\\imran\\Desktop\\gesture-vr-ai\\models\\gesture_model.onnx";
    
    // Initialise ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "gesture_inference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    Ort::Session session(env, model_path, session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    std::cout << "Model loaded successfully." << std::endl;
    std::cout << "Input node: landmarks (63 floats)" << std::endl;

    // --- Test with dummy data (all zeros = open hand probably) ---
    std::vector<float> dummy_landmarks(63, 0.0f);

    // Simulate a quick landmark test
    std::string result = predict_gesture(session, allocator, dummy_landmarks);
    std::cout << "Test prediction (dummy input): " << result << std::endl;

    // --- Interactive mode: paste 63 comma-separated floats ---
    std::cout << "\nEnter 63 comma-separated landmark values (or 'q' to quit):\n";

    std::string line;
    while (true) {
        std::cout << "> ";
        std::getline(std::cin, line);
        if (line == "q") break;

        std::vector<float> input;
        std::stringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ',')) {
            input.push_back(std::stof(token));
        }

        if (input.size() != 63) {
            std::cout << "Need exactly 63 values, got " << input.size() << std::endl;
            continue;
        }

        std::string gesture = predict_gesture(session, allocator, input);
        std::cout << "Predicted gesture: " << gesture << std::endl;
    }

    return 0;
}