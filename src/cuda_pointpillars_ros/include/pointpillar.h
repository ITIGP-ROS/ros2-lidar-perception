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
#include <memory>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include "NvInferRuntime.h"

#include "postprocess.h"
#include "preprocess.h"

// Per-stage timing. Each guarded block does a cudaDeviceSynchronize() or
// cudaEventSynchronize() plus a device-to-host cudaMemcpy.
//
// Measured cost, back-to-back on this bag: 21.00 ms/frame with it on vs
// 20.80 ms with it off -- about 1%, not the "half the frame rate" an earlier
// version of this comment claimed. The extra syncs buy little because the
// callback is already serialised around one blocking TensorRT call.
//
// It is OFF by default because three lines of stdout per frame is noise, not
// because it is expensive. Turn it on when profiling:
//
//     colcon build --packages-select cuda_pointpillars_ros \
//         --cmake-args -DCMAKE_BUILD_TYPE=Release -DPERFORMANCE_LOG=1
#ifndef PERFORMANCE_LOG
#define PERFORMANCE_LOG 0
#endif

// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level message
        //if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR || severity == Severity::kINFO ) {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "trt_infer: " << msg << std::endl;
        }
    }
};

class TRT {
  private:
    Params params_;

    cudaEvent_t start_, stop_;

    Logger gLogger_;
    nvinfer1::IExecutionContext *context_ = nullptr;
    nvinfer1::ICudaEngine *engine_ = nullptr;
    // TensorRT 10 requires the runtime to outlive any engine it deserialised,
    // so it is held here rather than being a local in the constructor.
    nvinfer1::IRuntime *runtime_ = nullptr;
    // I/O tensor names in binding order. TensorRT 10 addresses tensors by name
    // (enqueueV3) instead of by an index-ordered buffer array (enqueueV2).
    std::vector<std::string> io_names_;

    cudaStream_t stream_ = 0;
  public:
    TRT(std::string modelFile, cudaStream_t stream = 0);
    ~TRT(void);

    int doinfer(void**buffers);

    /// Shape of a named engine I/O tensor, so the caller can verify that the
    /// loaded engine actually agrees with params.h. Returns nbDims = -1 when
    /// the tensor does not exist.
    nvinfer1::Dims tensorShape(const char *name) const;
};

class PointPillar {
  private:
    Params params_;

    cudaEvent_t start_, stop_;
    cudaStream_t stream_;

    std::shared_ptr<PreProcessCuda> pre_;
    std::shared_ptr<TRT> trt_;
    std::shared_ptr<PostProcessCuda> post_;

    //input of pre-process
    float *voxel_features_ = nullptr;
    unsigned int *voxel_num_ = nullptr;
    unsigned int *voxel_idxs_ = nullptr;
    unsigned int *pillar_num_ = nullptr;

    unsigned int voxel_features_size_ = 0;
    unsigned int voxel_num_size_ = 0;
    unsigned int voxel_idxs_size_ = 0;

    //TRT-input
    float *features_input_ = nullptr;
    unsigned int *params_input_ = nullptr;
    unsigned int features_input_size_ = 0;

    //output of TRT -- input of post-process
    float *cls_output_ = nullptr;
    float *box_output_ = nullptr;
    float *dir_cls_output_ = nullptr;
    unsigned int cls_size_;
    unsigned int box_size_;
    unsigned int dir_cls_size_;

    //output of post-process
    float *bndbox_output_ = nullptr;
    unsigned int bndbox_size_ = 0;
    int max_bndbox_ = 0;          // capacity of bndbox_output_, in records

    std::vector<Bndbox> res_;

  public:
    PointPillar(std::string modelFile, cudaStream_t stream = 0);
    ~PointPillar(void);
    int doinfer(void*points, unsigned int point_size, std::vector<Bndbox> &res);

    nvinfer1::Dims tensorShape(const char *name) const { return trt_->tensorShape(name); }
};

