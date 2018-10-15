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

// Decodes the blocks generated by block_builder.cc.

#include "tensorflow/core/lib/io/block.h"

#include <algorithm>
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/format.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace table {

inline uint32 Block::NumRestarts() const {
  assert(size_ >= sizeof(uint32));
  return core::DecodeFixed32(data_ + size_ - sizeof(uint32));
}

Block::Block(const BlockContents& contents)
    : data_(contents.data.data()),
      size_(contents.data.size()),
      owned_(contents.heap_allocated) {
  if (size_ < sizeof(uint32)) {
    size_ = 0;  // Error marker
  } else {
    size_t max_restarts_allowed = (size_ - sizeof(uint32)) / sizeof(uint32);
    if (NumRestarts() > max_restarts_allowed) {
      // The size is too small for NumRestarts()
      size_ = 0;
    } else {
      restart_offset_ = size_ - (1 + NumRestarts()) * sizeof(uint32);
    }
  }
}

Block::~Block() {
  if (owned_) {
    delete[] data_;
  }
}

// Helper routine: decode the next block entry starting at "p",
// storing the number of shared key bytes, non_shared key bytes,
// and the length of the value in "*shared", "*non_shared", and
// "*value_length", respectively.  Will not dereference past "limit".
//
// If any errors are detected, returns NULL.  Otherwise, returns a
// pointer to the key delta (just past the three decoded values).
static inline const char* DecodeEntry(const char* p, const char* limit,
                                      uint32* shared, uint32* non_shared,
                                      uint32* value_length) {
  if (limit - p < 3) return nullptr;
  *shared = reinterpret_cast<const unsigned char*>(p)[0];
  *non_shared = reinterpret_cast<const unsigned char*>(p)[1];
  *value_length = reinterpret_cast<const unsigned char*>(p)[2];
  if ((*shared | *non_shared | *value_length) < 128) {
    // Fast path: all three values are encoded in one byte each
    p += 3;
  } else {
    if ((p = core::GetVarint32Ptr(p, limit, shared)) == nullptr) return nullptr;
    if ((p = core::GetVarint32Ptr(p, limit, non_shared)) == nullptr)
      return nullptr;
    if ((p = core::GetVarint32Ptr(p, limit, value_length)) == nullptr)
      return nullptr;
  }

  if (static_cast<uint32>(limit - p) < (*non_shared + *value_length)) {
    return nullptr;
  }
  return p;
}

class Block::Iter : public Iterator {
 private:
  const char* const data_;     // underlying block contents
  uint32 const restarts_;      // Offset of restart array (list of fixed32)
  uint32 const num_restarts_;  // Number of uint32 entries in restart array

  // current_ is offset in data_ of current entry.  >= restarts_ if !Valid
  uint32 current_;
  uint32 restart_index_;  // Index of restart block in which current_ falls
  string key_;
  absl::string_view value_;
  Status status_;

  inline int Compare(const absl::string_view& a,
                     const absl::string_view& b) const {
    return a.compare(b);
  }

  // Return the offset in data_ just past the end of the current entry.
  inline uint32 NextEntryOffset() const {
    return (value_.data() + value_.size()) - data_;
  }

  uint32 GetRestartPoint(uint32 index) {
    assert(index < num_restarts_);
    return core::DecodeFixed32(data_ + restarts_ + index * sizeof(uint32));
  }

  void SeekToRestartPoint(uint32 index) {
    key_.clear();
    restart_index_ = index;
    // current_ will be fixed by ParseNextKey();

    // ParseNextKey() starts at the end of value_, so set value_ accordingly
    uint32 offset = GetRestartPoint(index);
    value_ = absl::string_view(data_ + offset, 0);
  }

 public:
  Iter(const char* data, uint32 restarts, uint32 num_restarts)
      : data_(data),
        restarts_(restarts),
        num_restarts_(num_restarts),
        current_(restarts_),
        restart_index_(num_restarts_) {
    assert(num_restarts_ > 0);
  }

  bool Valid() const override { return current_ < restarts_; }
  Status status() const override { return status_; }
  absl::string_view key() const override {
    assert(Valid());
    return key_;
  }
  absl::string_view value() const override {
    assert(Valid());
    return value_;
  }

  void Next() override {
    assert(Valid());
    ParseNextKey();
  }

  void Seek(const absl::string_view& target) override {
    // Binary search in restart array to find the last restart point
    // with a key < target
    uint32 left = 0;
    uint32 right = num_restarts_ - 1;
    while (left < right) {
      uint32 mid = (left + right + 1) / 2;
      uint32 region_offset = GetRestartPoint(mid);
      uint32 shared, non_shared, value_length;
      const char* key_ptr =
          DecodeEntry(data_ + region_offset, data_ + restarts_, &shared,
                      &non_shared, &value_length);
      if (key_ptr == nullptr || (shared != 0)) {
        CorruptionError();
        return;
      }
      absl::string_view mid_key(key_ptr, non_shared);
      if (Compare(mid_key, target) < 0) {
        // Key at "mid" is smaller than "target".  Therefore all
        // blocks before "mid" are uninteresting.
        left = mid;
      } else {
        // Key at "mid" is >= "target".  Therefore all blocks at or
        // after "mid" are uninteresting.
        right = mid - 1;
      }
    }

    // Linear search (within restart block) for first key >= target
    SeekToRestartPoint(left);
    while (true) {
      if (!ParseNextKey()) {
        return;
      }
      if (Compare(key_, target) >= 0) {
        return;
      }
    }
  }

  void SeekToFirst() override {
    SeekToRestartPoint(0);
    ParseNextKey();
  }

 private:
  void CorruptionError() {
    current_ = restarts_;
    restart_index_ = num_restarts_;
    status_ = errors::DataLoss("bad entry in block");
    key_.clear();
    value_ = absl::string_view();
  }

  bool ParseNextKey() {
    current_ = NextEntryOffset();
    const char* p = data_ + current_;
    const char* limit = data_ + restarts_;  // Restarts come right after data
    if (p >= limit) {
      // No more entries to return.  Mark as invalid.
      current_ = restarts_;
      restart_index_ = num_restarts_;
      return false;
    }

    // Decode next entry
    uint32 shared, non_shared, value_length;
    p = DecodeEntry(p, limit, &shared, &non_shared, &value_length);
    if (p == nullptr || key_.size() < shared) {
      CorruptionError();
      return false;
    } else {
      key_.resize(shared);
      key_.append(p, non_shared);
      value_ = absl::string_view(p + non_shared, value_length);
      while (restart_index_ + 1 < num_restarts_ &&
             GetRestartPoint(restart_index_ + 1) < current_) {
        ++restart_index_;
      }
      return true;
    }
  }
};

Iterator* Block::NewIterator() {
  if (size_ < sizeof(uint32)) {
    return NewErrorIterator(errors::DataLoss("bad block contents"));
  }
  const uint32 num_restarts = NumRestarts();
  if (num_restarts == 0) {
    return NewEmptyIterator();
  } else {
    return new Iter(data_, restart_offset_, num_restarts);
  }
}

}  // namespace table
}  // namespace tensorflow
