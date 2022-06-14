/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/status.h"

#include <stdio.h>

#include <deque>
#include <functional>
#include <string>

#include "absl/base/call_once.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/types/optional.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/strcat.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/status.pb.h"

namespace tensorflow {

namespace {

// Log sink is used to collect recent warning and error log messages to be
// attached to the error status.
class StatusLogSink : public TFLogSink {
 public:
  static StatusLogSink* GetInstance() {
    static StatusLogSink* sink = new StatusLogSink();
    return sink;
  }

  void enable() {
    absl::call_once(flag_, [this] {
      num_messages_ = 5;  // default to 5 messages

      if (const char* num_msgs_str =
              getenv("TF_WORKER_NUM_FORWARDED_LOG_MESSAGES")) {
        if (!absl::SimpleAtoi(num_msgs_str, &num_messages_)) {
          LOG(WARNING) << "Failed to parse env variable "
                          "TF_WORKER_NUM_WARNING_ERROR_LOG_IN_STATUS="
                       << num_msgs_str << " as int. Using the default value "
                       << num_messages_ << ".";
        }
      }

      if (num_messages_ > 0) {
        TFAddLogSink(this);
      }
    });
  }

  void GetMessages(std::vector<std::string>* logs) TF_LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);

    for (auto& msg : messages_) {
      logs->push_back(msg);
    }
  }

  void Send(const TFLogEntry& entry) override TF_LOCKS_EXCLUDED(mu_) {
    if (entry.log_severity() < absl::LogSeverity::kWarning) return;

    mutex_lock lock(mu_);
    messages_.emplace_back(entry.ToString());
    if (messages_.size() > static_cast<size_t>(num_messages_)) {
      messages_.pop_front();
    }
  }

 private:
  mutex mu_;
  // for allowing repeated/concurrent calls to enable()
  absl::once_flag flag_;
  int num_messages_ = 0;
  std::deque<std::string> messages_ TF_GUARDED_BY(mu_);
};

}  // namespace

// TODO(b/197552541) Move this namespace to errors.h after absl migration.
namespace errors {
static constexpr const char kStackTraceProtoUrl[] =
    "type.googleapis.com/tensorflow.StackTracePayload";

void SetStackTrace(::tensorflow::Status& status,
                   std::vector<StackFrame> stack_trace) {
  StackTracePayload payload;
  for (const StackFrame& frame : stack_trace) {
    StackTracePayload::StackFrame frame_proto;
    frame_proto.set_file_name(frame.file_name);
    frame_proto.set_line_number(frame.line_number);
    frame_proto.set_function_name(frame.function_name);
    *payload.add_stack_frames() = frame_proto;
  }
  status.SetPayload(kStackTraceProtoUrl, payload.SerializeAsString());
}

std::vector<StackFrame> GetStackTrace(const ::tensorflow::Status& status) {
  std::vector<StackFrame> stack_trace;
  absl::optional<absl::string_view> maybe_serialized_payload =
      status.GetPayload(kStackTraceProtoUrl);
  if (maybe_serialized_payload.has_value()) {
    StackTracePayload payload;
    std::string serialized_payload(maybe_serialized_payload.value());
    payload.ParseFromString(serialized_payload);
    for (const auto& frame_proto : payload.stack_frames()) {
      StackFrame frame({frame_proto.file_name(), frame_proto.line_number(),
                        frame_proto.function_name()});
      stack_trace.push_back(frame);
    }
  }
  return stack_trace;
}

}  // namespace errors

Status::Status(tensorflow::error::Code code, absl::string_view msg) {
  assert(code != tensorflow::error::OK);
  state_ = std::unique_ptr<State>(new State);
  state_->code = code;
  state_->msg = std::string(msg);
  VLOG(5) << "Generated non-OK status: \"" << *this << "\". "
          << CurrentStackTrace();
}

void Status::Update(const Status& new_status) {
  if (ok()) {
    *this = new_status;
  }
}

void Status::SlowCopyFrom(const State* src) {
  if (src == nullptr) {
    state_ = nullptr;
  } else {
    state_ = std::unique_ptr<State>(new State(*src));
  }
}

Status OkStatus() { return Status(); }

const std::string& Status::empty_string() {
  static string* empty = new string;
  return *empty;
}

std::string error_name(error::Code code) {
  switch (code) {
    case tensorflow::error::OK:
      return "OK";
      break;
    case tensorflow::error::CANCELLED:
      return "CANCELLED";
      break;
    case tensorflow::error::UNKNOWN:
      return "UNKNOWN";
      break;
    case tensorflow::error::INVALID_ARGUMENT:
      return "INVALID_ARGUMENT";
      break;
    case tensorflow::error::DEADLINE_EXCEEDED:
      return "DEADLINE_EXCEEDED";
      break;
    case tensorflow::error::NOT_FOUND:
      return "NOT_FOUND";
      break;
    case tensorflow::error::ALREADY_EXISTS:
      return "ALREADY_EXISTS";
      break;
    case tensorflow::error::PERMISSION_DENIED:
      return "PERMISSION_DENIED";
      break;
    case tensorflow::error::UNAUTHENTICATED:
      return "UNAUTHENTICATED";
      break;
    case tensorflow::error::RESOURCE_EXHAUSTED:
      return "RESOURCE_EXHAUSTED";
      break;
    case tensorflow::error::FAILED_PRECONDITION:
      return "FAILED_PRECONDITION";
      break;
    case tensorflow::error::ABORTED:
      return "ABORTED";
      break;
    case tensorflow::error::OUT_OF_RANGE:
      return "OUT_OF_RANGE";
      break;
    case tensorflow::error::UNIMPLEMENTED:
      return "UNIMPLEMENTED";
      break;
    case tensorflow::error::INTERNAL:
      return "INTERNAL";
      break;
    case tensorflow::error::UNAVAILABLE:
      return "UNAVAILABLE";
      break;
    case tensorflow::error::DATA_LOSS:
      return "DATA_LOSS";
      break;
    default:
      char tmp[30];
      snprintf(tmp, sizeof(tmp), "UNKNOWN_CODE(%d)", static_cast<int>(code));
      return tmp;
      break;
  }
}

std::string Status::ToString() const {
  if (state_ == nullptr) {
    return "OK";
  } else {
    std::string result(error_name(state_->code));
    result += ": ";
    result += state_->msg;

    for (const std::pair<const std::string, std::string>& element :
         state_->payloads) {
      absl::StrAppend(&result, " [", element.first, "='",
                      absl::CHexEscape(element.second), "']");
    }

    return result;
  }
}

void Status::IgnoreError() const {
  // no-op
}

void Status::SetPayload(absl::string_view type_url, absl::string_view payload) {
  if (ok()) return;
  state_->payloads[std::string(type_url)] = std::string(payload);
}

absl::optional<absl::string_view> Status::GetPayload(
    absl::string_view type_url) const {
  if (ok()) return absl::nullopt;
  auto payload_iter = state_->payloads.find(std::string(type_url));
  if (payload_iter == state_->payloads.end()) return absl::nullopt;
  return absl::string_view(payload_iter->second);
}

bool Status::ErasePayload(absl::string_view type_url) {
  if (ok()) return false;
  auto payload_iter = state_->payloads.find(std::string(type_url));
  if (payload_iter == state_->payloads.end()) return false;
  state_->payloads.erase(payload_iter);
  return true;
}

void Status::ForEachPayload(
    const std::function<void(absl::string_view, absl::string_view)>& visitor)
    const {
  if (ok()) return;
  for (const auto& payload : state_->payloads) {
    visitor(payload.first, payload.second);
  }
}

std::ostream& operator<<(std::ostream& os, const Status& x) {
  os << x.ToString();
  return os;
}

Status AbortedError(absl::string_view message) {
  return Status(tensorflow::errors::Code::ABORTED, message);
}

Status AlreadyExistsError(absl::string_view message) {
  return Status(tensorflow::errors::Code::ALREADY_EXISTS, message);
}

Status CancelledError(absl::string_view message) {
  return Status(tensorflow::errors::Code::CANCELLED, message);
}

Status DataLossError(absl::string_view message) {
  return Status(tensorflow::errors::Code::DATA_LOSS, message);
}

Status DeadlineExceededError(absl::string_view message) {
  return Status(tensorflow::errors::Code::DEADLINE_EXCEEDED, message);
}

Status FailedPreconditionError(absl::string_view message) {
  return Status(tensorflow::errors::Code::FAILED_PRECONDITION, message);
}

Status InternalError(absl::string_view message) {
  return Status(tensorflow::errors::Code::INTERNAL, message);
}

Status InvalidArgumentError(absl::string_view message) {
  return Status(tensorflow::errors::Code::INVALID_ARGUMENT, message);
}

Status NotFoundError(absl::string_view message) {
  return Status(tensorflow::errors::Code::NOT_FOUND, message);
}

Status OutOfRangeError(absl::string_view message) {
  return Status(tensorflow::errors::Code::OUT_OF_RANGE, message);
}

Status PermissionDeniedError(absl::string_view message) {
  return Status(tensorflow::errors::Code::PERMISSION_DENIED, message);
}

Status ResourceExhaustedError(absl::string_view message) {
  return Status(tensorflow::errors::Code::RESOURCE_EXHAUSTED, message);
}

Status UnauthenticatedError(absl::string_view message) {
  return Status(tensorflow::errors::Code::UNAUTHENTICATED, message);
}

Status UnavailableError(absl::string_view message) {
  return Status(tensorflow::errors::Code::UNAVAILABLE, message);
}

Status UnimplementedError(absl::string_view message) {
  return Status(tensorflow::errors::Code::UNIMPLEMENTED, message);
}

Status UnknownError(absl::string_view message) {
  return Status(tensorflow::errors::Code::UNKNOWN, message);
}

std::string* TfCheckOpHelperOutOfLine(const ::tensorflow::Status& v,
                                      const char* msg) {
  std::string r("Non-OK-status: ");
  r += msg;
  r += " status: ";
  r += v.ToString();
  // Leaks string but this is only to be used in a fatal error message
  return new std::string(r);
}

StatusGroup::StatusGroup() {}

StatusGroup::StatusGroup(std::initializer_list<Status> statuses) {
  for (const Status& s : statuses) {
    Update(s);
  }
}

static constexpr const char kDerivedStatusProtoUrl[] =
    "type.googleapis.com/tensorflow.DerivedStatus";

Status StatusGroup::MakeDerived(const Status& s) {
  if (IsDerived(s)) {
    return s;
  } else {
    Status derived(s);
    // TODO(b/200167936): Serialize an instance of DerivedStatus proto instead
    // of using the string directly. The string is never used so it is not
    // causing any issues at the moment.
    derived.SetPayload(kDerivedStatusProtoUrl, "");
    return derived;
  }
}

bool StatusGroup::IsDerived(const Status& s) {
  return s.GetPayload(kDerivedStatusProtoUrl).has_value();
}

void StatusGroup::ConfigureLogHistory() {
  StatusLogSink::GetInstance()->enable();
}

void StatusGroup::Update(const Status& s) {
  if (s.ok()) {
    ++num_ok_;
  } else {
    ok_ = false;
    if (IsDerived(s)) {
      derived_.insert(s);
    } else {
      non_derived_.insert(s);
    }
  }
}

static constexpr int kMaxAggregatedStatusMessageSize = 8 * 1024;
static constexpr int kMaxAttachedLogMessageSize = 512;

std::unordered_map<std::string, std::string> StatusGroup::GetPayloads() const {
  std::unordered_map<std::string, std::string> payloads;
  auto capture_payload = [&payloads](absl::string_view key,
                                     absl::string_view value) {
    payloads[std::string(key)] = std::string(value);
  };

  for (const auto& status : derived_) {
    status.ForEachPayload(capture_payload);
  }

  // If a key appears in both derived_ and non_derived_ payloads, then the
  // non_derived_ payload receives priority.
  for (const auto& status : non_derived_) {
    status.ForEachPayload(capture_payload);
  }

  payloads.erase(kDerivedStatusProtoUrl);

  return payloads;
}

Status MakeStatus(
    tensorflow::error::Code code, absl::string_view message,
    const std::unordered_map<std::string, std::string>& payloads) {
  Status status(code, message);
  for (const auto& payload : payloads) {
    status.SetPayload(payload.first, payload.second);
  }
  return status;
}

std::string MakeString(const Status& status) {
  return absl::StrCat(error_name(status.code()), ": ", status.error_message());
}

// Summarize all the status objects in the StatusGroup. This is used when
// individual Status objects in the StatusGroup are not already summarized.
Status StatusGroup::as_summary_status() const {
  if (ok_) {
    return Status::OK();
  }

  // Gather recent logs as a string
  auto get_recent_logs = [this]() -> std::string {
    if (!recent_logs_.empty()) {
      std::vector<std::string> fmt;
      fmt.push_back("\nRecent warning and error logs:");
      for (auto& log : recent_logs_) {
        // Add an indentation to make it look nicer.
        fmt.push_back("  " + log.substr(0, kMaxAttachedLogMessageSize));
      }
      return absl::StrJoin(fmt, "\n");
    } else {
      return "";
    }
  };

  // If only one root status is found, do not add summary header and footer.
  if (non_derived_.size() == 1) {
    return MakeStatus(non_derived_.begin()->code(),
                      strings::StrCat(non_derived_.begin()->error_message(),
                                      get_recent_logs()),
                      GetPayloads());
  }

  if (!non_derived_.empty()) {
    std::vector<std::string> fmt;

    fmt.push_back(
        strings::Printf("%zu root error(s) found.", non_derived_.size()));

    int index = 0;
    auto code = tensorflow::error::CANCELLED;
    for (const auto& s : non_derived_) {
      // NOTE: Avoid using CANCELLED as the code of summary status if the group
      // contains other error code.
      if (code == tensorflow::error::CANCELLED &&
          s.code() != tensorflow::error::CANCELLED) {
        code = s.code();
      }
      fmt.emplace_back(strings::StrCat("  (", index, ") ", MakeString(s)));
      ++index;
    }

    fmt.push_back(strings::Printf("%zu successful operations.", num_ok_));
    fmt.push_back(
        strings::Printf("%zu derived errors ignored.", derived_.size()));

    std::string error_msg =
        absl::StrJoin(fmt, "\n").substr(0, kMaxAggregatedStatusMessageSize);

    return MakeStatus(code, strings::StrCat(error_msg, get_recent_logs()),
                      GetPayloads());
  } else {
    // All statuses are derived. Pick the first available status to return.
    return MakeDerived(MakeStatus(derived_.begin()->code(),
                                  derived_.begin()->error_message(),
                                  GetPayloads()));
  }
}

// Concatenate all the status objects in the StatusGroup. This is used when
// individual Status objects in the StatusGroup are already summarized Status.
Status StatusGroup::as_concatenated_status() const {
  if (ok_) {
    return Status::OK();
  }

  // If only one root status is found, return it directly.
  if (non_derived_.size() == 1) {
    return MakeStatus(non_derived_.begin()->code(),
                      non_derived_.begin()->error_message(), GetPayloads());
  }

  if (!non_derived_.empty()) {
    std::vector<string> fmt;
    fmt.emplace_back("\n=====================");
    for (const auto& s : non_derived_) {
      fmt.emplace_back(MakeString(s));
    }
    fmt.emplace_back("=====================\n");
    return MakeStatus(
        non_derived_.begin()->code(),
        absl::StrJoin(fmt, "\n").substr(0, kMaxAggregatedStatusMessageSize),
        GetPayloads());
  } else {
    // All statuses are derived. Pick the first available status to return.
    // This should not happen in normal execution.
    return MakeDerived(MakeStatus(derived_.begin()->code(),
                                  derived_.begin()->error_message(),
                                  GetPayloads()));
  }
}

void StatusGroup::AttachLogMessages() {
  recent_logs_.clear();
  StatusLogSink::GetInstance()->GetMessages(&recent_logs_);
}

}  // namespace tensorflow
