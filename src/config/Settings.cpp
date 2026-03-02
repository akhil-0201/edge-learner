#include "config/Settings.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace el {

std::string Settings::getEnv(const char* key, const char* def) {
  const char* v = std::getenv(key);
  return v ? std::string(v) : std::string(def);
}

int Settings::getEnvInt(const char* key, int def) {
  const char* v = std::getenv(key);
  return v ? std::stoi(v) : def;
}

float Settings::getEnvFloat(const char* key, float def) {
  const char* v = std::getenv(key);
  return v ? std::stof(v) : def;
}

void Settings::loadClassNames() {
  classNames.clear();
  if (classNamesPath.empty()) return;
  std::ifstream f(classNamesPath);
  if (!f.is_open()) return;
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty()) classNames.push_back(line);
  }
}

Settings Settings::fromEnv() {
  Settings s;
  s.modelPath              = getEnv("MODEL_PATH", "/opt/edge-learner/models/current.rknn");
  s.modelVersion           = getEnv("MODEL_VERSION", "v1.0.0");
  s.variantId              = getEnv("VARIANT_ID", "default");
  s.numCores               = getEnvInt("NPU_CORE_COUNT", 3);
  s.confidenceThreshold    = getEnvFloat("CONFIDENCE_THRESHOLD", 0.5f);
  s.lowConfThreshold       = getEnvFloat("LOW_CONF_THRESHOLD", 0.3f);
  s.iouThreshold           = getEnvFloat("IOU_THRESHOLD", 0.45f);
  s.inputSize              = getEnvInt("INPUT_SIZE", 640);
  s.classNamesPath         = getEnv("CLASS_NAMES_PATH", "/opt/edge-learner/models/classes.txt");
  s.cameraSource           = getEnv("CAMERA_SOURCE", "0");
  s.cameraWidth            = getEnvInt("CAMERA_WIDTH", 1280);
  s.cameraHeight           = getEnvInt("CAMERA_HEIGHT", 720);
  s.cameraFps              = getEnvInt("CAMERA_FPS", 30);
  s.bufferDir              = getEnv("BUFFER_DIR", "/opt/edge-learner/buffer");
  s.bufferMaxSize          = getEnvInt("BUFFER_MAX_SIZE", 500);
  s.jpegQuality            = getEnvInt("JPEG_QUALITY", 85);
  s.trainingTriggerCount   = getEnvInt("TRAINING_TRIGGER_COUNT", 100);
  s.trainingEpochs         = getEnvInt("TRAINING_EPOCHS", 50);
  s.trainingBatchSize      = getEnvInt("TRAINING_BATCH_SIZE", 16);
  s.baseModelPath          = getEnv("BASE_MODEL_PATH", "/opt/edge-learner/models/base.pt");
  s.trainingDataDir        = getEnv("TRAINING_DATA_DIR", "/opt/edge-learner/training_data");
  s.exportDir              = getEnv("EXPORT_DIR", "/opt/edge-learner/exports");
  s.pythonExe              = getEnv("PYTHON_EXE", "/opt/edge-learner/venv/bin/python");
  s.trainerScript          = getEnv("TRAINER_SCRIPT", "/opt/edge-learner/trainer/fine_tuner.py");
  s.uploadEndpoint         = getEnv("UPLOAD_ENDPOINT", "");
  s.uploadApiKey           = getEnv("UPLOAD_API_KEY", "");
  s.uploadTimeout          = getEnvInt("UPLOAD_TIMEOUT", 120);
  s.logLevel               = getEnv("LOG_LEVEL", "INFO");
  s.logDir                 = getEnv("LOG_DIR", "/opt/edge-learner/logs");

  // Create required directories
  for (const auto& d : {s.bufferDir, s.trainingDataDir, s.exportDir, s.logDir}) {
    std::filesystem::create_directories(d);
  }

  s.loadClassNames();
  return s;
}

Settings Settings::fromYaml(const std::string& path) {
  // Start with env defaults then overlay YAML
  Settings s = fromEnv();
  YAML::Node cfg = YAML::LoadFile(path);

  auto str = [&](const char* k, std::string& v) {
    if (cfg[k]) v = cfg[k].as<std::string>();
  };
  auto i = [&](const char* k, int& v) {
    if (cfg[k]) v = cfg[k].as<int>();
  };
  auto f = [&](const char* k, float& v) {
    if (cfg[k]) v = cfg[k].as<float>();
  };

  str("model_path",           s.modelPath);
  str("model_version",        s.modelVersion);
  str("variant_id",           s.variantId);
  i  ("npu_core_count",       s.numCores);
  f  ("confidence_threshold", s.confidenceThreshold);
  f  ("low_conf_threshold",   s.lowConfThreshold);
  f  ("iou_threshold",        s.iouThreshold);
  i  ("input_size",           s.inputSize);
  str("class_names_path",     s.classNamesPath);
  str("camera_source",        s.cameraSource);
  i  ("camera_width",         s.cameraWidth);
  i  ("camera_height",        s.cameraHeight);
  i  ("camera_fps",           s.cameraFps);
  str("buffer_dir",           s.bufferDir);
  i  ("buffer_max_size",      s.bufferMaxSize);
  i  ("jpeg_quality",         s.jpegQuality);
  i  ("training_trigger_count", s.trainingTriggerCount);
  i  ("training_epochs",      s.trainingEpochs);
  i  ("training_batch_size",  s.trainingBatchSize);
  str("base_model_path",      s.baseModelPath);
  str("training_data_dir",    s.trainingDataDir);
  str("export_dir",           s.exportDir);
  str("python_exe",           s.pythonExe);
  str("trainer_script",       s.trainerScript);
  str("upload_endpoint",      s.uploadEndpoint);
  str("upload_api_key",       s.uploadApiKey);
  i  ("upload_timeout",       s.uploadTimeout);
  str("log_level",            s.logLevel);
  str("log_dir",              s.logDir);

  s.loadClassNames();
  return s;
}

} // namespace el
