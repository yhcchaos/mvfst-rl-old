/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#include "CongestionControlEnv.h"
#include <iostream>
#include <folly/Conv.h>
#include <quic/congestion_control/CongestionControlFunctions.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/shm.h>
#include <csignal>

namespace quic {

using Field = NetworkState::Field;

static const float kBytesToMB = 1.0 / 1024 / 1024;
// CongestionControlEnv impl
//actor shared memory to exchange infomation among actors to compute rewards
int64_t shm_id_actor = -1;
float* shm_addr_actor = nullptr;
//link shared memory to pass link bandwidth to actor
int64_t shm_id_link = -1;
float* shm_addr_link = nullptr;
void sighandler(int sig){
  std::cerr << "receive SIGTERM signal, exit and release shared memory" 
            << ". shm_id_actor: " << shm_id_actor << ", shm_addr_actor: " << shm_addr_actor 
            << ", shm_id_link: "  << shm_id_link  << ", shm_addr_link: " << shm_addr_link 
            << std::endl;
  if(shmdt((void*)shm_addr_actor)==-1){
    perror("shmdt_actor:");
    std::cerr <<"shmdt_actor errno:"<< errno << std::endl;
  }
  if(shmctl(shm_id_actor, IPC_RMID, 0)==-1){
    perror("shmctl_rm_actor:");
    std::cerr <<"shmctl_rm_actor errno:"<< errno << std::endl;
  }
  if(shmdt((void*)shm_addr_link)==-1){
    perror("shmdt_link:");
    std::cerr <<"shmdt_link errno:"<< errno << std::endl;
  }
  if(shmctl(shm_id_link, IPC_RMID, 0)==-1){
    perror("shmctl_rm_Link:");
    std::cerr <<"shmctl_rm_link errno:"<< errno << std::endl;
  }
  exit(1);
}

CongestionControlEnv::CongestionControlEnv(const Config& cfg, Callback* cob,
                                           const QuicConnectionStateBase& conn)
    : cfg_(cfg),
      cob_(CHECK_NOTNULL(cob)),
      conn_(conn),
      evb_(folly::EventBaseManager::get()->getEventBase()),
      observationTimeout_(this, evb_),
      cwndBytes_(conn.transportSettings.initCwndInMss * conn.udpSendPacketLen) {
  // Initialize history with no-op past actions
  History noopHistory(0.0);
  history_.resize(cfg.historySize, noopHistory);
  lastTdp_ = 10 * conn_.udpSendPacketLen;
  std::signal(SIGTERM, sighandler);
  std::signal(SIGKILL, sighandler);
  shm_id_actor = shmget((cfg_.actorId << 24) | cfg_.episode_id, sizeof(float) * cfg_.flows, IPC_CREAT | 0666);
  shm_addr_actor = (float*)shmat(shm_id_actor, 0, 0);
  shm_id_link = shmget((cfg_.actorId << 28) | cfg_.episode_id, 2*sizeof(float), IPC_CREAT | 0666);
  shm_addr_link = (float*)shmat(shm_id_link, 0, 0);

  if(shm_id_actor == -1 || shm_addr_actor == nullptr || shm_id_link == -1 || shm_addr_actor == nullptr){
    int64_t pgid = getpgid(getpid());
    VLOG(1) << "没有获得共享链路带宽或actor的内存句柄，或共享内存关联失败,杀死该环境中所有流的进程";
    kill(-pgid, SIGTERM);
  } 
  /*std::cerr << shm_key << " " << shm_id << " " <<  cfg_.flows << " " << cfg_.flowId << std::endl;
  if(shm_id == -1){
    perror("shm create error( ");
  }
  */
  
  if (cfg.aggregation == Config::Aggregation::TIME_WINDOW) {
    CHECK_GT(cfg.windowDuration.count(), 0);
    observationTimeout_.schedule(cfg.windowDuration);
  }
}

CongestionControlEnv::~CongestionControlEnv(){
  shmdt((void*)shm_addr_actor);
  shmctl(shm_id_actor, IPC_RMID, 0);
  shmdt((void*)shm_addr_link);
  shmctl(shm_id_link, IPC_RMID, 0);

}
void CongestionControlEnv::onAction(const Action& action) {
  evb_->runImmediatelyOrRunInEventBaseThreadAndWait([this, action] {
    float prevCwndBytes = cwndBytes_;
    updateCwnd(action.cwndAction);
    cob_->onUpdate(cwndBytes_);

    // Update history
    history_.pop_front();
    history_.emplace_back(cwndBytes_ / prevCwndBytes - 1);

    const auto& elapsed = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - lastActionTime_);
    lastActionTime_ = std::chrono::steady_clock::now();
    /*
    VLOG(1) << "Action updated (cwndAction=" << action.cwndAction
            << ", cwnd=" << cwndBytes_ / conn_.udpSendPacketLen
            << "), policy elapsed time = " << elapsed.count() << " ms";
    */
  });
}

void CongestionControlEnv::onNetworkState(NetworkState&& state) {
  VLOG(3) << __func__ << ": " << state;

  states_.push_back(std::move(state));

  switch (cfg_.aggregation) {
    case Config::Aggregation::TIME_WINDOW:
      DCHECK(observationTimeout_.isScheduled());
      break;
    case Config::Aggregation::FIXED_WINDOW:
      if (states_.size() == cfg_.windowSize) {
        handleStates();
      }
      break;
    default:
      LOG(FATAL) << "Unknown aggregation type";
      break;
  }
}

void CongestionControlEnv::observationTimeoutExpired() noexcept {
  handleStates();
  observationTimeout_.schedule(cfg_.windowDuration);
}

void CongestionControlEnv::handleStates() {
  if (states_.empty()) {
    return;
  }

  // Compute reward based on original states
  const auto& rewards = computeReward(states_);

  Observation obs(cfg_);
  obs.states = useStateSummary() ? stateSummary(states_) : std::move(states_);
  states_.clear();
  //{bw_reward, delay_reward, loss_reward, reward};
  obs.rewards = rewards;
  obs.env.resize(4);

  obs.env[0]=shm_addr_link[1] / 10;
  obs.env[1]=cfg_.delay / 10;
  obs.env[2]=cfg_.lossRatio;
  obs.env[3]=cfg_.flows;
  std::copy(history_.begin(), history_.end(), std::back_inserter(obs.history));

  VLOG(2) << __func__ << ' ' << obs;

  lastObservationTime_ = std::chrono::steady_clock::now();

  float episodeWeight = 1.0 / float(cfg_.flows);
  onObservation(std::move(obs), rewards[4], episodeWeight);
}

std::vector<NetworkState> CongestionControlEnv::stateSummary(
    const std::vector<NetworkState>& states) {
  int dim = 0;
  bool keepdim = true;
  // Bassel's correction on stddev only when defined to avoid NaNs.
  bool unbiased = (states.size() > 1);

  NetworkState::toTensor(states, summaryTensor_);
  const auto& std_mean =
      torch::std_mean(summaryTensor_, dim, unbiased, keepdim);
  const auto& summaryMean = std::get<1>(std_mean);
  const auto& summary = torch::cat(
      {summaryMean, torch::sub(summaryMean, lastSummaryMean_)}, dim);
  lastSummaryMean_ = summaryMean;
  auto summaryStates = NetworkState::fromTensor(summary);

  static const std::vector<std::string> keys = {
      "Mean", "Delta",
  };
  VLOG(2) << "State summary: ";
  for (size_t i = 0; i < summaryStates.size(); ++i) {
    VLOG(2) << keys[i] << ": " << summaryStates[i];
  }

  return summaryStates;
}

std::vector<float> CongestionControlEnv::computeReward(
    const std::vector<NetworkState>& states) {
  // Reward function is a combinaton of throughput, delay and lost bytes.
  // For throughput and delay, it makes sense to take the average, whereas
  // for loss, we compute the total bytes lost over these states.
  float avgGoodput = 0.0;
  float avgLrtt = 0.0;
  float totalLost = 0.0;
  for (const auto& state : states) {
    avgGoodput += state[Field::ACKED];
    avgLrtt += state[Field::LRTT];
    totalLost += state[Field::LOST];
  }
  avgLrtt = states.size() > 0 ? avgLrtt / states.size() : cfg_.delay * 2;
  const auto& elapsed = std::chrono::duration<float, std::milli>(
        std::chrono::steady_clock::now() - lastObservationTime_);

  lastTdp_ = avgGoodput * normBytes() / elapsed.count() * avgLrtt * normMs();

  // Undo normalization and convert to Mbps for throughput and ms for
  // delay.
  float goodputMbps = avgGoodput * normBytes() * kBytesToMB / elapsed.count() * 1000 * 8;
  float avgLrttMs = avgLrtt * normMs();
  // Ignore cfg_.maxDelayInReward for now.
  float avgQDelayMs = avgLrttMs - cfg_.delay * 2;
  float lossMbps = totalLost * normBytes() * kBytesToMB / elapsed.count() * 1000 * 8;
  float changeTime = shm_addr_link[0];
  float bandwidth = shm_addr_link[1];
  float throughputFactor = cfg_.throughputFactor;
  if(bandwidth>=150){
    throughputFactor = 1.5;
  }

  float bw_share = bandwidth / float(cfg_.flows);
  float exceed_sum = 0;
  float reg = 1;
  float exceed_bw = std::max(goodputMbps - bw_share, float(0));
  shm_addr_actor[cfg_.flowId - 1] = exceed_bw;
  for(uint32_t i=0;i<cfg_.flows;i++){
    //std::cerr << this->shm_addr[i] << " ";
    if(shm_addr_actor[i] > 0){
      exceed_sum += shm_addr_actor[i];
    }
  }
  //std::cerr << std::endl;
  float shared_error = (exceed_bw == 0 ? 0 : exceed_bw / exceed_sum);
  //VLOG(1) << "exceed_sum= " << exceed_sum << ", " << "shared_error = " << shared_error << std::endl;
  reg = shared_error * cfg_.flows;
  if(cfg_.flows == 1){
    reg = 1;
  }
  //VLOG(1) << "reg= " << reg;
  float bw_reward = throughputFactor * std::min(goodputMbps / bandwidth * float(cfg_.flows), float(1));
  float delay_reward = avgQDelayMs;
  float delay_reward_all =  cfg_.delayFactor * delay_reward * reg;
  float loss_reward =  cfg_.packetLossFactor * (lossMbps - cfg_.lossRatio / (1 - cfg_.lossRatio) * goodputMbps) / bandwidth * float(cfg_.flows);
  float reward = bw_reward - delay_reward_all - loss_reward;
  if(times % 100 == 0){
    VLOG(1) << "change time=" << changeTime
            << ", change bw=" << bandwidth
            << ", flows=" << cfg_.flows 
            << ", flow Id=" << cfg_.flowId
            << ", gpid=" << getpgid(getpid())
            << ", bw_reward=" << bw_reward
            << ", delay_reward=" << avgQDelayMs
            << ", avg throughput=" << goodputMbps
            << ", bw_share=" << bw_share
            << ", exceed_self=" <<  exceed_bw
            << ", exceed_sum=" << exceed_sum
            << ", shared_error=" << shared_error
            << ", reg=" << reg
            << ", delay_reward_all= " << delay_reward_all
            << ", actor id=" << cfg_.actorId
            << ", episode id=" << cfg_.episode_id
            << ", flow Id=" << cfg_.flowId
            << ", Num states=" << states.size()
            << " avg LRTT=" << avgLrttMs
            << " loss=" << lossMbps << " Mbps, reward = " << reward
            << " state elapsed time = " << elapsed.count() << " ms";
  }
  times+=1;
  return {bw_reward, delay_reward, delay_reward_all, loss_reward, reward};
}

void CongestionControlEnv::updateCwnd(const uint32_t actionIdx) {
  DCHECK_LT(actionIdx, cfg_.actions.size());
  const auto& op = cfg_.actions[actionIdx].first;
  const auto& val = cfg_.actions[actionIdx].second;
  const auto& valBytes = val * conn_.udpSendPacketLen;

  switch (op) {
    case Config::ActionOp::NOOP:
      break;
    case Config::ActionOp::ADD:
      cwndBytes_ += valBytes;
      break;
    case Config::ActionOp::SUB:
      cwndBytes_ = (cwndBytes_ >= valBytes) ? (cwndBytes_ - valBytes) : 0;
      break;
    case Config::ActionOp::MUL:
      cwndBytes_ = std::round(cwndBytes_ * val);
      break;
    case Config::ActionOp::DIV:
      cwndBytes_ = std::round(cwndBytes_ * 1.0 / val);
      break;
    default:
      LOG(FATAL) << "Unknown ActionOp";
      break;
  }

  cwndBytes_ = std::min(cwndBytes_, std::max(lastTdp_ * 5, uint64_t(100)));
  cwndBytes_ = boundedCwnd(cwndBytes_, conn_.udpSendPacketLen,
                           conn_.transportSettings.maxCwndInMss,
                           conn_.transportSettings.minCwndInMss);
}

/// CongestionControlEnv::Observation impl

torch::Tensor CongestionControlEnv::Observation::toTensor() const {
  torch::Tensor tensor = torch::empty({0}, torch::kFloat32);
  toTensor(tensor);
  return tensor;
}

void CongestionControlEnv::Observation::toTensor(torch::Tensor& tensor) const {
  if (states.empty()) {
    tensor.resize_({0});
    return;
  }

  CHECK_EQ(history.size(), cfg_.historySize);

  // Total dim = flattened state dim + history dim.
  uint32_t dim = states.size() * states[0].size() + history.size() + 9;

  tensor.resize_({dim});
  auto tensor_a = tensor.accessor<float, 1>();
  int x = 0;

  // Serialize states
  for (const auto& state : states) {
    for (size_t i = 0; i < state.size(); ++i) {
      tensor_a[x++] = state[i];
    }
  }

  // Serialize history
  for (const auto& h : history) {
    tensor_a[x++] = h.action;
  }
  
  for(int i=0;i<5;i++){
    tensor_a[x++] = rewards[i];
  }

  for(int i=0;i<4;i++){
    tensor_a[x++] = env[i];
  }
  CHECK_EQ(x, dim);
}

std::ostream& operator<<(std::ostream& os,
                         const CongestionControlEnv::Observation& obs) {
  os << "Observation (" << obs.states.size() << " states, "
     << obs.history.size() << " history):" << std::endl;
  for (const auto& state : obs.states) {
    os << state << std::endl;
  }
  for (const auto& history : obs.history) {
    os << history << std::endl;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os,
                         const CongestionControlEnv::History& history) {
  os << "History: action=" << history.action;
  return os;
}

}  // namespace quic