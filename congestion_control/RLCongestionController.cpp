/*
* Copyright (c) Facebook, Inc. and its affiliates.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*
*/
#include <quic/common/TimeUtil.h>
#include <quic/congestion_control/CongestionControlFunctions.h>
#include <quic/congestion_control/Copa.h>

#include "NetworkState.h"
#include "RLCongestionController.h"

namespace quic {

using namespace std::chrono;

using Field = NetworkState::Field;

RLCongestionController::RLCongestionController(
    QuicConnectionStateBase& conn,
    std::shared_ptr<CongestionControlEnvFactory> envFactory)
    : conn_(conn),
      cwndBytes_(conn.transportSettings.initCwndInMss * conn.udpSendPacketLen),
      env_(envFactory->make(this, conn)),
      minRTTFilter_(kMinRTTWindowLength.count(), 0us, 0),
      standingRTTFilter_(100000, /*100ms*/
                         0us, 0) {
  VLOG(10) << __func__ << " writable=" << getWritableBytes()
           << " cwnd=" << cwndBytes_ << " inflight=" << bytesInFlight_ << " "
           << conn_;
}

void RLCongestionController::onRemoveBytesFromInflight(uint64_t bytes) {
  subtractAndCheckUnderflow(bytesInFlight_, bytes);
  VLOG(10) << __func__ << " writable=" << getWritableBytes()
           << " cwnd=" << cwndBytes_ << " inflight=" << bytesInFlight_ << " "
           << conn_;
}

void RLCongestionController::onPacketSent(const OutstandingPacketWrapper& packet) {
  addAndCheckOverflow(bytesInFlight_, packet.metadata.encodedSize);

  VLOG(10) << __func__ << " writable=" << getWritableBytes()
           << " cwnd=" << cwndBytes_ << " inflight=" << bytesInFlight_
           << " bytesBufferred=" << conn_.flowControlState.sumCurStreamBufferLen
           << " packetNum=" << packet.packet.header.getPacketSequenceNum()
           << " " << conn_;
}

void RLCongestionController::onPacketAckOrLoss(const AckEvent* FOLLY_NULLABLE ack,
                      const LossEvent* FOLLY_NULLABLE loss){
  if (loss) {
    onPacketLoss(*loss);
  }
  if (ack) {
    onPacketAcked(*ack);
  }

  // State update to the env
  NetworkState obs;
  if (setNetworkState(ack, loss, obs)) {
    env_->onNetworkState(std::move(obs));
  }
}

void RLCongestionController::onPacketAcked(const AckEvent& ack) {
  subtractAndCheckUnderflow(bytesInFlight_, ack.ackedBytes);
  minRTTFilter_.Update(
      conn_.lossState.lrtt,
      std::chrono::duration_cast<microseconds>(ack.ackTime.time_since_epoch())
          .count());
  standingRTTFilter_.SetWindowLength(conn_.lossState.srtt.count() / 2);
  standingRTTFilter_.Update(
      conn_.lossState.lrtt,
      std::chrono::duration_cast<microseconds>(ack.ackTime.time_since_epoch())
          .count());

  VLOG(10) << __func__ << "ack size=" << ack.ackedBytes
           << " num packets acked=" << ack.ackedBytes / conn_.udpSendPacketLen
           << " writable=" << getWritableBytes() << " cwnd=" << cwndBytes_
           << " inflight=" << bytesInFlight_
           << " sRTT=" << conn_.lossState.srtt.count()
           << " lRTT=" << conn_.lossState.lrtt.count()
           << " mRTT=" << conn_.lossState.mrtt.count()
           << " rttvar=" << conn_.lossState.rttvar.count()
           << " packetsBufferred="
           << conn_.flowControlState.sumCurStreamBufferLen
           << " packetsRetransmitted=" << conn_.lossState.rtxCount << " "
           << conn_;
}

void RLCongestionController::onPacketLoss(const LossEvent& loss) {
  VLOG(10) << __func__ << " lostBytes=" << loss.lostBytes
           << " lostPackets=" << loss.lostPackets << " cwnd=" << cwndBytes_
           << " inflight=" << bytesInFlight_ << " " << conn_;
  DCHECK(loss.largestLostPacketNum.hasValue());
  subtractAndCheckUnderflow(bytesInFlight_, loss.lostBytes);
  if (loss.persistentCongestion) {
    VLOG(10) << __func__ << " writable=" << getWritableBytes()
             << " cwnd=" << cwndBytes_ << " inflight=" << bytesInFlight_ << " "
             << conn_;
  }
}

void RLCongestionController::onUpdate(const uint64_t& cwndBytes) noexcept {
  cwndBytes_ = cwndBytes;
  if(conn_.pacer){
    conn_.pacer->refreshPacingRate(cwndBytes_, conn_.lossState.srtt);
  }
}

bool RLCongestionController::setNetworkState(
    const AckEvent* FOLLY_NULLABLE ack,
    const LossEvent* FOLLY_NULLABLE loss,
    NetworkState& obs) {
  const auto& state = conn_.lossState;

  const auto& rttMin = minRTTFilter_.GetBest();
  const auto& rttStanding = standingRTTFilter_.GetBest().count();
  const auto& delay = rttStanding - rttMin.count();
      // duration_cast<microseconds>(conn_.lossState.lrtt - rttMin).count();
  if (rttStanding == 0 || delay < 0) {
    LOG(ERROR)
        << "Invalid rttStanding or delay, skipping network state update: "
        << "rttStanding = " << rttStanding << ", delay = " << delay << " "
        << conn_;
    return false;
  }

  const float normMs = env_->normMs();
  const float normBytes = env_->normBytes();

  obs[Field::RTT_MIN] = rttMin.count() / 1000.0 / normMs;
  obs[Field::RTT_STANDING] = rttStanding / 1000.0 / normMs;
  obs[Field::LRTT] = state.lrtt.count() / 1000.0 / normMs;
  obs[Field::SRTT] = state.srtt.count() / 1000.0 / normMs;
  obs[Field::RTT_VAR] = state.rttvar.count() / 1000.0 / normMs;
  obs[Field::DELAY] = delay / 1000.0 / normMs;
  //和损失里面的throughput有什么区别，throughput=AckedPacketsSize/(AckReceiveTime-1stAckedPacketSentTime)
  obs[Field::THROUGHPUT] = 1.0 / obs[Field::SRTT];
  obs[Field::IN_FLIGHT] = float(bytesInFlight_) / float(cwndBytes_);
  obs[Field::WRITABLE] = float(getWritableBytes()) / float(cwndBytes_);
  obs[Field::SENT] = (state.totalBytesSent - prevTotalBytesSent_) / normBytes;
  obs[Field::RECEIVED] =
      (state.totalBytesRecvd - prevTotalBytesRecvd_) / normBytes;
  obs[Field::RETRANSMITTED] =
      (state.totalBytesRetransmitted - prevTotalBytesRetransmitted_) /
      normBytes;
  if(ack){
    obs[Field::ACKED] = ack->ackedBytes / normBytes;
  }
  if(loss){
    obs[Field::LOST] = loss->lostBytes / normBytes;
  }
  
  // Update prev state values
  prevTotalBytesSent_ = state.totalBytesSent;
  prevTotalBytesRecvd_ = state.totalBytesRecvd;
  prevTotalBytesRetransmitted_ = state.totalBytesRetransmitted;
  prevTotalPTOCount_ = state.totalPTOCount;
  prevRtxCount_ = state.rtxCount;
  prevTimeoutBasedRtxCount_ = state.timeoutBasedRtxCount;

  return true;
}

uint64_t RLCongestionController::getWritableBytes() const noexcept {
  if (bytesInFlight_ > cwndBytes_) {
    return 0;
  } else {
    return cwndBytes_ - bytesInFlight_;
  }
}

uint64_t RLCongestionController::getCongestionWindow() const noexcept {
  return cwndBytes_;
}

CongestionControlType RLCongestionController::type() const noexcept {
  return CongestionControlType::None;
}

uint64_t RLCongestionController::getBytesInFlight() const noexcept {
  return bytesInFlight_;
}

void RLCongestionController::setAppIdle(bool,
                                        TimePoint) noexcept { /* unsupported */
}

void RLCongestionController::setAppLimited() { /* unsupported */
}

bool RLCongestionController::isAppLimited() const noexcept {
  return false;  // not supported
}
void RLCongestionController::setBandwidthUtilizationFactor(
      float bandwidthUtilizationFactor) noexcept {}
bool RLCongestionController::isInBackgroundMode() const noexcept {
  return false;
}
void RLCongestionController:: getStats(CongestionControllerStats& stats) const{

}
}  // namespace quic
