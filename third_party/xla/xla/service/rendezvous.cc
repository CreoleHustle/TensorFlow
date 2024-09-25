/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/rendezvous.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string_view>

#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace internal {

void AwaitAndLogIfStuck(std::atomic<int32_t>& ack, absl::Notification& ready,
                        std::string_view name, size_t num_threads,
                        absl::Duration warn_stuck_timeout,
                        absl::Duration terminate_timeout) {
  if (ready.WaitForNotificationWithTimeout(warn_stuck_timeout)) {
    return;
  }

  // Check if all rendezvous participants arrived to the rendezvous point and
  // incremented `ack` counter. We still can be stuck because the leader is
  // waiting for completion of rendezvous callback, but it must not be confused
  // with participants not arriving to the rendezvous point.
  bool is_all_participants_arrived = ack.load() == num_threads;

  if (is_all_participants_arrived) {
    LOG(ERROR) << absl::StreamFormat(
        "This thread has been waiting for `%s` for %d seconds and may be "
        "stuck. All %d threads joined the rendezvous, however the leader has "
        "not marked the rendezvous as completed. Leader can be deadlocked "
        "inside the rendezvous callback.",
        name, absl::ToInt64Seconds(warn_stuck_timeout), num_threads);

  } else {
    LOG(ERROR) << absl::StreamFormat(
        "This thread has been waiting for `%s` for %d seconds and may be "
        "stuck. Expected %d threads to join the rendezvous, but not all of "
        "them arrived on time.",
        name, absl::ToInt64Seconds(warn_stuck_timeout), num_threads);
  }

  if (ready.WaitForNotificationWithTimeout(terminate_timeout)) {
    LOG(ERROR) << "Thread is unstuck! Warning above was a false-positive. "
                  "Perhaps the timeout is too short.";
    return;
  }

  if (is_all_participants_arrived) {
    LOG(FATAL) << absl::StreamFormat(
        "Termination timeout for `%s` of %d seconds exceeded. Exiting to "
        "ensure a consistent program state. All %d threads joined the "
        "rendezvous, however the leader has not marked the rendezvous as "
        "completed. Leader can be deadlocked inside the rendezvous callback.",
        name, absl::ToInt64Seconds(terminate_timeout), num_threads);

  } else {
    LOG(FATAL) << absl::StreamFormat(
        "Termination timeout for `%s` of %d seconds exceeded. Exiting to "
        "ensure a consistent program state. Expected %d threads to join the "
        "rendezvous, but not all of them arrived on time.",
        name, absl::ToInt64Seconds(terminate_timeout), num_threads);
  }
}

}  // namespace internal

namespace {
inline constexpr int32_t kPending = 0;
inline constexpr int32_t kCompleted = std::numeric_limits<int32_t>::max();
}  // namespace

RendezvousSingleFlag::RendezvousSingleFlag() : state_(kPending) {}

RendezvousSingleFlag::InFlightRendezvous::InFlightRendezvous(
    RendezvousSingleFlag* flag)
    : flag_(flag) {}

RendezvousSingleFlag::InFlightRendezvous::~InFlightRendezvous() {
  if (flag_ == nullptr) return;

  // Reload state and use CAS to decide if we are the one who
  // should mark rendezvous flag completed.
  int32_t state = flag_->state_.load();

  CHECK(state != kPending && state != kCompleted)  // NOLINT
      << "rendezvous can't be in pending or completed state";

  // Exit the critical section and maybe mark rendezvous as completed.
  while (!flag_->state_.compare_exchange_weak(
      state, state == 1 ? kCompleted : state - 1)) {
    // Check state after CAS failure: while we are in this function no one
    // should complete rendezvous without us or switch it back to pending.
    CHECK(state != kPending && state != kCompleted);  // NOLINT
  }
}

RendezvousSingleFlag::InFlightRendezvous::operator bool() const {
  return flag_ != nullptr;
}

RendezvousSingleFlag::InFlightRendezvous RendezvousSingleFlag::TryJoin() {
  // If `state_` is `kCompleted` it means that we have at least one completed
  // rendezvous for this flag and can skip it.
  if (state_.load() == kCompleted) return InFlightRendezvous(nullptr);

  // Try to increment a state in a CAS loop to signal all other participants
  // that we joined an in-flight rendezvous.
  int32_t state = state_.load();
  while (state != kCompleted &&
         !state_.compare_exchange_weak(state, state + 1)) {
  }

  // Someone else completed the rendezvous and we don't need to join.
  if (state == kCompleted) return InFlightRendezvous(nullptr);

  return InFlightRendezvous(this);
}

bool RendezvousSingleFlag::IsCompleted() const {
  return state_.load() == kCompleted;
}

}  // namespace xla
