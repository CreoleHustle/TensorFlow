/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <fcntl.h>
#include <getopt.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/optional_debug_tools.h"
#include "tensorflow/contrib/lite/string_util.h"

#include "label_image.h"

#define LOG(x) std::cerr
#define CHECK(x)                  \
  if (!(x)) {                     \
    LOG(ERROR) << #x << "failed"; \
    exit(1);                      \
  }

namespace tflite {
namespace label_image {

using std::string;

bool verbose = false;
bool accel = true;
bool input_floating = false;
int loop_count = 1;

float input_mean = 127.5f;
float input_std = 127.5f;

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
TfLiteStatus ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(FATAL) << "Labels file " << file_name << " not found\n";
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

void RunInference(const std::string& graph, const std::string& input_layer_type,
                  int num_threads, const std::string& input_bmp_name,
                  const std::string& labels_file_name) {
  CHECK(graph.c_str());

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;

  model = tflite::FlatBufferModel::BuildFromFile(graph.c_str());
  if (!model) {
    LOG(FATAL) << "\nFailed to mmap model " << graph << "\n";
    exit(-1);
  }
  LOG(INFO) << "Loaded model " << graph << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";

  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }

  if (!accel) interpreter->UseNNAPI(accel);

  if (verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
    LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
    LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point << "\n";
    }
  }

  if (num_threads != -1) {
    interpreter->SetNumThreads(num_threads);
  }

  int image_width = 224;
  int image_height = 224;
  int image_channels = 3;
  uint8_t* in =
      read_bmp(input_bmp_name, image_width, image_height, image_channels);

  int input = interpreter->inputs()[0];
  if (verbose) LOG(INFO) << "input: " << input << "\n";

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  if (verbose) {
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

  if (verbose) PrintInterpreterState(interpreter.get());

  // get input dimension from the input tensor metadata
  // assuming one input only
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];

  if (input_floating)
    downsize<float>(interpreter->typed_tensor<float>(input), in, image_height,
                    image_width, image_channels, wanted_height, wanted_width,
                    wanted_channels);
  else
    downsize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in,
                      image_height, image_width, image_channels, wanted_height,
                      wanted_width, wanted_channels);

  struct timeval start_time, stop_time;
  gettimeofday(&start_time, NULL);
  for (int i = 0; i < loop_count; i++) {
    if (interpreter->Invoke() != kTfLiteOk) {
      LOG(FATAL) << "Failed to invoke tflite!\n";
    }
  }
  gettimeofday(&stop_time, NULL);
  LOG(INFO) << "invoked \n";
  LOG(INFO) << "average time: "
            << (get_us(stop_time) - get_us(start_time)) / (loop_count * 1000)
            << " ms \n";

  const int output_size = 1000;
  const size_t num_results = 5;
  const float threshold = 0.001f;

  std::vector<std::pair<float, int>> top_results;

  if (input_floating)
    get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                     num_results, threshold, &top_results);
  else
    get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                       output_size, num_results, threshold, &top_results);

  std::vector<string> labels;
  size_t label_count;

  if (ReadLabelsFile(labels_file_name, &labels, &label_count) != kTfLiteOk)
    exit(-1);

  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;
    LOG(INFO) << confidence << ": " << index << " " << labels[index] << "\n";
  }
}

void display_usage() {
  LOG(INFO) << "label_image\n"
            << "--accelerated, -a: [0|1], use Android NNAPI or note\n"
            << "--count, -c: loop interpreter->Invoke() for certain times\n"
            << "--input_floating, -f: [0|1] type of input layer is floating "
               "point numbers\n"
            << "--input_mean, -b: input mean\n"
            << "--input_std, -s: input standard deviation\n"
            << "--image, -i: image_name.bmp\n"
            << "--labels, -l: labels for the model\n"
            << "--tflite_mode, -m: model_name.tflite\n"
            << "--threads, -t: number of threads\n"
            << "--verbose, -v: [0|1] print more information\n"
            << "\n";
}

int Main(int argc, char** argv) {
  // std::string model_name = "/data/local/tmp/mobilenet_quant_v1_224.tflite";
  string model_name = "./mobilenet_quant_v1_224.tflite";
  string input_bmp_name = "./grace_hopper.bmp";
  string labels_file_name = "./labels.txt";
  string input_layer_type = "uint8_t";
  int number_of_threads = 4;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"accelerated", required_argument, 0, 'a'},
        {"count", required_argument, 0, 'c'},
        {"input_floating", required_argument, 0, 'f'},
        {"verbose", required_argument, 0, 'v'},
        {"image", required_argument, 0, 'i'},
        {"labels", required_argument, 0, 'l'},
        {"tflite_model", required_argument, 0, 'm'},
        {"threads", required_argument, 0, 't'},
        {"input_mean", required_argument, 0, 'b'},
        {"input_std", required_argument, 0, 's'},
        {0, 0, 0, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "a:b:c:f:i:l:m:s:t:v:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        accel = optarg;
        break;
      case 'b':
        input_mean = strtod(optarg, NULL);
        break;
      case 'c':
        loop_count = strtol(optarg, (char**)NULL, 10);
        break;
      case 'f':
        input_floating = strtol(optarg, (char**)NULL, 10);
        input_layer_type = "float";
        break;
      case 'i':
        input_bmp_name = optarg;
        break;
      case 'l':
        labels_file_name = optarg;
        break;
      case 'm':
        model_name = optarg;
        break;
      case 's':
        input_std = strtod(optarg, NULL);
        break;
      case 't':
        number_of_threads = strtol(optarg, (char**)NULL, 10);
        break;
      case 'v':
        verbose = strtol(optarg, (char**)NULL, 10);
        break;
      case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
      default:
        exit(-1);
    }
  }
  RunInference(model_name, input_layer_type, number_of_threads, input_bmp_name,
               labels_file_name);
  return 0;
}

}  // namespace label_image
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::label_image::Main(argc, argv);
}
