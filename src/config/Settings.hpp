#pragma once
#include <string>
#include <vector>
#include <cstdlib>

namespace el {

struct Settings {
  // Model
  std::string modelPath;
  std::string modelVersion;
  std::string variantId;
  int         numCores;

  // Inference
  float       confidenceThreshold;
  float       lowConfThreshold;
  float       iouThreshold;
  int         inputSize;
  std::string classNamesPath;

  // Camera
  std::string cameraSource;
  int         cameraWidth;
  int         cameraHeight;
  int         cameraFps;

  // Buffer
  std::string bufferDir;
  int         bufferMaxSize;
  int         jpegQuality;

  // Trainer
  int         trainingTriggerCount;
  int         trainingEpochs;
  int         trainingBatchSize;
  std::string baseModelPath;
  std::string trainingDataDir;
  std::string exportDir;
  std::string pythonExe;
  std::string trainerScript;

  // Uploader
  std::string uploadEndpoint;
  std::string uploadApiKey;
  int         uploadTimeout;

  // Logging
  std::string logLevel;
  std::string logDir;

  // Parsed class names (loaded from file)
  std::vector<std::string> classNames;

  static Settings fromEnv();
  static Settings fromYaml(const std::string& path);

private:
  void loadClassNames();
  static std::string getEnv(const char* key, const char* def);
  static int         getEnvInt(const char* key, int def);
  static float       getEnvFloat(const char* key, float def);
};

} // namespace el
