/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pointpillar.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"

TRT::~TRT(void)
{
  delete(context_);
  delete(engine_);
  delete(runtime_);
  checkCudaErrors(cudaEventDestroy(start_));
  checkCudaErrors(cudaEventDestroy(stop_));
  return;
}

namespace {

// A serialised TensorRT engine is valid ONLY for the exact TensorRT version,
// GPU architecture and build precision that produced it -- and, obviously, for
// the model it was built from. The original code keyed the cache purely on the
// filename ("<model>.onnx.cache"), so swapping in a different .onnx while an
// old cache was present would silently keep running the PREVIOUS engine: no
// error, quietly wrong results. That bites the moment the KITTI model is
// swapped for the Livox one. Fold all four facts into the filename instead.
std::string engineCachePath(const std::string &modelFile)
{
  // FNV-1a over the ONNX contents, so editing or replacing the model invalidates.
  uint64_t hash = 1469598103934665603ULL;
  std::ifstream in(modelFile, std::ios::binary);
  std::vector<char> buf(64 * 1024);
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const std::streamsize n = in.gcount();
    if (n <= 0) break;
    for (std::streamsize i = 0; i < n; ++i) {
      hash ^= static_cast<unsigned char>(buf[i]);
      hash *= 1099511628211ULL;
    }
  }

  cudaDeviceProp prop{};
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);

  std::ostringstream os;
  os << modelFile << '.' << std::hex << hash << std::dec
     << ".trt" << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR
     << ".sm" << prop.major << prop.minor
#if defined (__arm64__) || defined (__aarch64__)
     << ".fp16"
#else
     << ".fp32"
#endif
     << ".engine";
  return os.str();
}

}  // namespace

TRT::TRT(std::string modelFile, cudaStream_t stream):stream_(stream)
{
  std::string modelCache = engineCachePath(modelFile);
  std::cout << "TRT engine cache: " << modelCache << std::endl;
  std::fstream trtCache(modelCache, std::ifstream::in);
  checkCudaErrors(cudaEventCreate(&start_));
  checkCudaErrors(cudaEventCreate(&stop_));
  if (!trtCache.is_open())
  {
	  std::cout << "Building TRT engine."<<std::endl;
    // define builder
    auto builder = (nvinfer1::createInferBuilder(gLogger_));

    // define network
#if NV_TENSORRT_MAJOR >= 10
    // TensorRT 10 removed kEXPLICIT_BATCH; explicit batch is the only mode.
    auto network = (builder->createNetworkV2(0));
#else
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = (builder->createNetworkV2(explicitBatch));
#endif

    // define onnxparser
    auto parser = (nvonnxparser::createParser(*network, gLogger_));
    if (!parser->parseFromFile(modelFile.data(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        std::cerr << ": failed to parse onnx model file, please check the onnx version and trt support op!"
                  << std::endl;
        exit(-1);
    }

    // define config
    auto networkConfig = builder->createBuilderConfig();
#if defined (__arm64__) || defined (__aarch64__) 
    networkConfig->setFlag(nvinfer1::BuilderFlag::kFP16);
    std::cout << "Enable fp16!" << std::endl;
#endif
#if NV_TENSORRT_MAJOR >= 10
    // setMaxBatchSize and setMaxWorkspaceSize were both removed in TRT 10.
    networkConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, size_t(1) << 30);

    // buildEngineWithConfig is gone; build a serialised network and deserialise.
    auto trtModelStream = builder->buildSerializedNetwork(*network, *networkConfig);
    if (trtModelStream == nullptr)
    {
      std::cerr << ": failed to build serialized engine!" << std::endl;
      exit(-1);
    }
    runtime_ = nvinfer1::createInferRuntime(gLogger_);
    engine_ = runtime_->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size());
    if (engine_ == nullptr)
    {
      std::cerr << ": engine init null!" << std::endl;
      exit(-1);
    }
#else
    // set max batch size
    builder->setMaxBatchSize(1);
    // set max workspace
    networkConfig->setMaxWorkspaceSize(size_t(1) << 30);

    engine_ = (builder->buildEngineWithConfig(*network, *networkConfig));

    if (engine_ == nullptr)
    {
      std::cerr << ": engine init null!" << std::endl;
      exit(-1);
    }

    // serialize the engine, then close everything down
    auto trtModelStream = (engine_->serialize());
#endif
    std::fstream trtOut(modelCache, std::ifstream::out);
    if (!trtOut.is_open())
    {
       std::cout << "Can't store trt cache.\n";
       exit(-1);
    }

    trtOut.write((char*)trtModelStream->data(), trtModelStream->size());
    trtOut.close();
#if NV_TENSORRT_MAJOR >= 10
    // destroy() was removed in TRT 10; these are ordinary deletable objects.
    delete trtModelStream;
    delete networkConfig;
    delete parser;
    delete network;
    delete builder;
#else
    trtModelStream->destroy();

    networkConfig->destroy();
    parser->destroy();
    network->destroy();
    builder->destroy();
#endif

  } else {
	  // std::cout << "load TRT cache."<<std::endl;
    char *data;
    unsigned int length;

    // get length of file:
    trtCache.seekg(0, trtCache.end);
    length = trtCache.tellg();
    trtCache.seekg(0, trtCache.beg);

    data = (char *)malloc(length);
    if (data == NULL ) {
       std::cout << "Can't malloc data.\n";
       exit(-1);
    }

    trtCache.read(data, length);
    // create context. Held as a member: TensorRT 10 requires the runtime to
    // outlive the engine it produced, and the original local went out of scope.
    runtime_ = nvinfer1::createInferRuntime(gLogger_);

    if (runtime_ == nullptr) {
        std::cerr << ": runtime null!" << std::endl;
        exit(-1);
    }
#if NV_TENSORRT_MAJOR >= 10
    // The trailing plugin-factory argument was removed in TRT 10.
    engine_ = (runtime_->deserializeCudaEngine(data, length));
#else
    engine_ = (runtime_->deserializeCudaEngine(data, length, 0));
#endif
    if (engine_ == nullptr) {
        std::cerr << ": engine null!" << std::endl;
        exit(-1);
    }
    free(data);
    trtCache.close();
  }

  context_ = engine_->createExecutionContext();

#if NV_TENSORRT_MAJOR >= 10
  // Cache the I/O tensor names in binding order so doinfer() can bind the
  // caller's buffer array by name. Expected order, from the ONNX graph:
  //   voxels, voxel_idxs, voxel_num, cls_preds, box_preds, dir_cls_preds
  const int nb_io = engine_->getNbIOTensors();
  io_names_.reserve(nb_io);
  for (int i = 0; i < nb_io; ++i) {
    io_names_.emplace_back(engine_->getIOTensorName(i));
  }
#endif
  return;
}

nvinfer1::Dims TRT::tensorShape(const char *name) const
{
  nvinfer1::Dims none{};
  none.nbDims = -1;
  if (engine_ == nullptr) {
    return none;
  }
#if NV_TENSORRT_MAJOR >= 10
  return engine_->getTensorShape(name);
#else
  const int idx = engine_->getBindingIndex(name);
  if (idx < 0) {
    return none;
  }
  return engine_->getBindingDimensions(idx);
#endif
}

int TRT::doinfer(void**buffers)
{
#if NV_TENSORRT_MAJOR >= 10
  // TensorRT 10 replaced the index-ordered buffer array of enqueueV2 with
  // per-tensor addresses set by name. io_names_ is in binding order, so the
  // caller's array maps across positionally and the call site is unchanged.
  for (size_t i = 0; i < io_names_.size(); ++i) {
    if (!context_->setTensorAddress(io_names_[i].c_str(), buffers[i])) {
      std::cerr << "trt_infer: failed to bind tensor " << io_names_[i] << std::endl;
      return -1;
    }
  }
  if (!context_->enqueueV3(stream_)) {
      return -1;
  }
#else
  int status;

  status = context_->enqueueV2(buffers, stream_, &start_);

  if (!status)
  {
      return -1;
  }
#endif

  return 0;
}

PointPillar::PointPillar(std::string modelFile, cudaStream_t stream):stream_(stream)
{
  checkCudaErrors(cudaEventCreate(&start_));
  checkCudaErrors(cudaEventCreate(&stop_));

  pre_.reset(new PreProcessCuda(stream_));
  trt_.reset(new TRT(modelFile, stream_));
  post_.reset(new PostProcessCuda(stream_));

  //point cloud to voxels
  voxel_features_size_ = MAX_VOXELS * params_.max_num_points_per_pillar * 4 * sizeof(float);
  voxel_num_size_ = MAX_VOXELS * sizeof(unsigned int);
  voxel_idxs_size_ = MAX_VOXELS* 4 * sizeof(unsigned int);

  checkCudaErrors(cudaMallocManaged((void **)&voxel_features_, voxel_features_size_));
  checkCudaErrors(cudaMallocManaged((void **)&voxel_num_, voxel_num_size_));
  checkCudaErrors(cudaMallocManaged((void **)&voxel_idxs_, voxel_idxs_size_));

  checkCudaErrors(cudaMemsetAsync(voxel_features_, 0, voxel_features_size_, stream_));
  checkCudaErrors(cudaMemsetAsync(voxel_num_, 0, voxel_num_size_, stream_));
  checkCudaErrors(cudaMemsetAsync(voxel_idxs_, 0, voxel_idxs_size_, stream_));

  //TRT-input
  features_input_size_ = MAX_VOXELS * params_.max_num_points_per_pillar * 10 * sizeof(float);
  checkCudaErrors(cudaMallocManaged((void **)&features_input_, features_input_size_));
  checkCudaErrors(cudaMallocManaged((void **)&params_input_, sizeof(unsigned int)));

  checkCudaErrors(cudaMemsetAsync(features_input_, 0, features_input_size_, stream_));
  checkCudaErrors(cudaMemsetAsync(params_input_, 0, sizeof(unsigned int), stream_));

  //output of TRT -- input of post-process
  cls_size_ = params_.feature_x_size * params_.feature_y_size * params_.num_classes * params_.num_anchors * sizeof(float);
  box_size_ = params_.feature_x_size * params_.feature_y_size * params_.num_box_values * params_.num_anchors * sizeof(float);
  dir_cls_size_ = params_.feature_x_size * params_.feature_y_size * params_.num_dir_bins * params_.num_anchors * sizeof(float);
  checkCudaErrors(cudaMallocManaged((void **)&cls_output_, cls_size_));
  checkCudaErrors(cudaMallocManaged((void **)&box_output_, box_size_));
  checkCudaErrors(cudaMallocManaged((void **)&dir_cls_output_, dir_cls_size_));

  //output of post-process
  max_bndbox_ = params_.feature_x_size * params_.feature_y_size * params_.num_anchors;
  bndbox_size_ = static_cast<size_t>(max_bndbox_) * kBndboxStride * sizeof(float);
  checkCudaErrors(cudaMallocManaged((void **)&bndbox_output_, bndbox_size_));

  res_.reserve(100);
  return;
}

PointPillar::~PointPillar(void)
{
  pre_.reset();
  trt_.reset();
  post_.reset();

  checkCudaErrors(cudaFree(voxel_features_));
  checkCudaErrors(cudaFree(voxel_num_));
  checkCudaErrors(cudaFree(voxel_idxs_));

  checkCudaErrors(cudaFree(features_input_));
  checkCudaErrors(cudaFree(params_input_));

  checkCudaErrors(cudaFree(cls_output_));
  checkCudaErrors(cudaFree(box_output_));
  checkCudaErrors(cudaFree(dir_cls_output_));

  checkCudaErrors(cudaFree(bndbox_output_));

  checkCudaErrors(cudaEventDestroy(start_));
  checkCudaErrors(cudaEventDestroy(stop_));
  return;
}

int PointPillar::doinfer(void*points_data, unsigned int points_size, std::vector<Bndbox> &nms_pred)
{
#if PERFORMANCE_LOG
  float generateVoxelsTime = 0.0f;
  checkCudaErrors(cudaEventRecord(start_, stream_));
#endif

  pre_->generateVoxels((float*)points_data, points_size,
        params_input_,
        voxel_features_, 
        voxel_num_,
        voxel_idxs_);

#if PERFORMANCE_LOG
  checkCudaErrors(cudaEventRecord(stop_, stream_));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventElapsedTime(&generateVoxelsTime, start_, stop_));
  unsigned int params_input_cpu;
  checkCudaErrors(cudaMemcpy(&params_input_cpu, params_input_, sizeof(unsigned int), cudaMemcpyDefault));
  if (getenv("PP_DEBUG_PILLARS")) {
    // One-shot dump of what preprocessing actually produced, so the pillar
    // features can be diffed against a reference implementation on the same
    // cloud. This is the bisect between "preprocessing is wrong" and "decode
    // is wrong" -- everything else looks correct on inspection.
    static int frame = 0;
    checkCudaErrors(cudaStreamSynchronize(stream_));
    std::cout << "[pp_debug] frame " << frame << " pillars=" << params_input_cpu << std::endl;
    if (frame == 0 && params_input_cpu > 0) {
      const unsigned int np = std::min(params_input_cpu, 3u);
      std::vector<float> feat(np * params_.max_num_points_per_pillar * 4);
      checkCudaErrors(cudaMemcpy(feat.data(), voxel_features_,
                                 feat.size() * sizeof(float), cudaMemcpyDefault));
      std::vector<unsigned int> idxs(np * 4), nums(np);
      checkCudaErrors(cudaMemcpy(idxs.data(), voxel_idxs_, idxs.size() * sizeof(unsigned int), cudaMemcpyDefault));
      checkCudaErrors(cudaMemcpy(nums.data(), voxel_num_, nums.size() * sizeof(unsigned int), cudaMemcpyDefault));
      for (unsigned int p = 0; p < np; ++p) {
        std::cout << "[pp_debug] pillar " << p << " npoints=" << nums[p]
                  << " idx=(" << idxs[p*4+0] << "," << idxs[p*4+1] << ","
                  << idxs[p*4+2] << "," << idxs[p*4+3] << ") first_point_xyzi=";
        for (int f = 0; f < 4; ++f) {
          std::cout << feat[p * params_.max_num_points_per_pillar * 4 + f] << " ";
        }
        std::cout << std::endl;
      }
    }
    ++frame;
  }
#endif

#if PERFORMANCE_LOG
  float generateFeaturesTime = 0.0f;
  checkCudaErrors(cudaEventRecord(start_, stream_));
#endif

  pre_->generateFeatures(voxel_features_,
      voxel_num_,
      voxel_idxs_,
      params_input_,
      features_input_);

#if PERFORMANCE_LOG
  checkCudaErrors(cudaEventRecord(stop_, stream_));
  checkCudaErrors(cudaEventSynchronize(stop_));
  checkCudaErrors(cudaEventElapsedTime(&generateFeaturesTime, start_, stop_));
#endif

#if PERFORMANCE_LOG
  float doinferTime = 0.0f;
  checkCudaErrors(cudaEventRecord(start_, stream_));
#endif

  void *buffers[] = {features_input_, voxel_idxs_, params_input_, cls_output_, box_output_, dir_cls_output_};
  trt_->doinfer(buffers);
  if (getenv("PP_DUMP_CLS")) {
    // One-shot dump of the raw network outputs, so they can be diffed against
    // the PyTorch reference on the same cloud. This is the bisect between
    // "inference/preprocessing differs" and "decode differs".
    static bool done = false;
    if (!done) {
      done = true;
      checkCudaErrors(cudaStreamSynchronize(stream_));
      auto dump = [&](const char *name, float *dptr, size_t n) {
        std::vector<float> h(n);
        checkCudaErrors(cudaMemcpy(h.data(), dptr, n * sizeof(float), cudaMemcpyDefault));
        FILE *f = fopen((std::string(getenv("PP_DUMP_CLS")) + "/" + name).c_str(), "wb");
        fwrite(h.data(), sizeof(float), n, f);
        fclose(f);
        std::cout << "[pp_dump] " << name << " n=" << n << std::endl;
      };
      dump("cls.bin", cls_output_, cls_size_ / sizeof(float));
      dump("box.bin", box_output_, box_size_ / sizeof(float));
      dump("dir.bin", dir_cls_output_, dir_cls_size_ / sizeof(float));
    }
  }
  checkCudaErrors(cudaMemsetAsync(params_input_, 0, sizeof(unsigned int), stream_));

#if PERFORMANCE_LOG
  checkCudaErrors(cudaEventRecord(stop_, stream_));
  checkCudaErrors(cudaEventSynchronize(stop_));
  checkCudaErrors(cudaEventElapsedTime(&doinferTime, start_, stop_));
#endif

#if PERFORMANCE_LOG
  float doPostprocessCudaTime = 0.0f;
  checkCudaErrors(cudaEventRecord(start_, stream_));
#endif

  int num_obj = 0;
  post_->doPostprocessCuda(cls_output_, box_output_, dir_cls_output_,
                          bndbox_output_, max_bndbox_, &num_obj);
  checkCudaErrors(cudaDeviceSynchronize());

  const float *output = bndbox_output_;

  // The kernel emits every (anchor, class) pair above the score threshold --
  // ~350 per frame at 0.1, out of 321408 anchors. The reference decode instead
  // keeps only the top `nms_pre` ANCHORS by max class score before it does
  // anything else, then caps the final list at `max_num`. Without both caps
  // this pipeline emitted 44.4 boxes/frame against the reference's 19.9.
  // Ranking is by anchor, not by box, so the two boxes an anchor can emit for
  // two different classes stand or fall together -- hence anchor_index in the
  // record.
  std::vector<int> anchor_of(num_obj);
  res_.reserve(num_obj);
  for (int i = 0; i < num_obj; i++) {
    const float *r = output + i * kBndboxStride;
    res_.emplace_back(r[0], r[1], r[2], r[3], r[4], r[5], r[6],
                      static_cast<int>(r[7]), r[8]);
    anchor_of[i] = static_cast<int>(r[9]);
  }

  if (params_.nms_pre > 0 && num_obj > 0) {
    // Group the records by anchor. Only a few hundred boxes clear the score
    // threshold, so sorting indices is far cheaper than a lookup table over all
    // feature_x * feature_y * num_anchors positions.
    std::vector<int> idx(num_obj);
    for (int i = 0; i < num_obj; i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b) { return anchor_of[a] < anchor_of[b]; });

    struct Group { float best; int first, last; };   // [first, last) into idx
    std::vector<Group> groups;
    for (int i = 0; i < num_obj;) {
      int j = i;
      float best = 0.0f;
      while (j < num_obj && anchor_of[idx[j]] == anchor_of[idx[i]]) {
        best = std::max(best, res_[idx[j]].score);
        j++;
      }
      groups.push_back({best, i, j});
      i = j;
    }

    if (static_cast<int>(groups.size()) > params_.nms_pre) {
      std::nth_element(groups.begin(), groups.begin() + params_.nms_pre, groups.end(),
                       [](const Group &a, const Group &b) { return a.best > b.best; });
      groups.resize(params_.nms_pre);
      std::vector<Bndbox> kept;
      kept.reserve(params_.nms_pre * params_.num_classes);
      for (const Group &g : groups) {
        for (int k = g.first; k < g.last; k++) kept.push_back(res_[idx[k]]);
      }
      res_.swap(kept);
    }
  }

  nms_cpu(res_, params_.nms_thresh, nms_pred);
  res_.clear();

  if (params_.max_num > 0 && static_cast<int>(nms_pred.size()) > params_.max_num) {
    std::partial_sort(nms_pred.begin(), nms_pred.begin() + params_.max_num, nms_pred.end(),
                      [](const Bndbox &a, const Bndbox &b) { return a.score > b.score; });
    nms_pred.resize(params_.max_num);
  }

#if PERFORMANCE_LOG
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(stop_, stream_));
  checkCudaErrors(cudaEventSynchronize(stop_));
  checkCudaErrors(cudaEventElapsedTime(&doPostprocessCudaTime, start_, stop_));
  std::cout<<"TIME: generateVoxels: "<< generateVoxelsTime <<" ms." <<std::endl;
  std::cout<<"TIME: generateFeatures: "<< generateFeaturesTime <<" ms." <<std::endl;
  std::cout<<"TIME: doinfer: "<< doinferTime <<" ms." <<std::endl;
  // std::cout<<"TIME: doPostprocessCuda: "<< doPostprocessCudaTime <<" ms." <<std::endl;
#endif
  return 0;
}
