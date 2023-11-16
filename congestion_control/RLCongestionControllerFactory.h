/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#pragma once

#include <glog/logging.h>
#include <quic/QuicConstants.h>
#include <quic/congestion_control/CongestionControllerFactory.h>

#include <memory>

#include "RLCongestionController.h"

namespace quic {

struct CongestionController;
struct QuicConnectionStateBase;

class RLCongestionControllerFactory : public CongestionControllerFactory {
 public:
  RLCongestionControllerFactory(
      std::shared_ptr<CongestionControlEnvFactory> envFactory, int64_t* shm_id_addr_, void** shm_addr_addr_)
      : envFactory_(CHECK_NOTNULL(envFactory)), shm_id_addr(shm_id_addr_), shm_addr_addr(shm_addr_addr_) {}

  ~RLCongestionControllerFactory() override = default;

  std::unique_ptr<CongestionController> makeCongestionController(
      QuicConnectionStateBase& conn, CongestionControlType type) {
    LOG(INFO) << "Creating RLCongestionController";
    conn.transportSettings.pacingEnabled = true;
    return std::make_unique<RLCongestionController>(conn, envFactory_, shm_id_addr, shm_addr_addr);
  }

 private:
  std::shared_ptr<CongestionControlEnvFactory> envFactory_;
  int64_t* shm_id_addr{nullptr}; 
  void** shm_addr_addr{nullptr};
};

}  // namespace quic
