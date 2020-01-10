// The utility to read checkpoints for google brain tensor ops and v3
// checkpoints for dist_belief.
//

#ifndef TENSORFLOW_UTIL_TENSOR_SLICE_READER_H_
#define TENSORFLOW_UTIL_TENSOR_SLICE_READER_H_

#include <unordered_map>

#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "tensorflow/core/util/saved_tensor_slice.pb.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/tensor_slice_set.h"
#include "tensorflow/core/util/tensor_slice_util.h"

namespace tensorflow {

namespace checkpoint {

// The reader reads in all the meta data about all the tensor slices. Then it
// will try to read the relevant data on-demand to produce the data for the
// slices needed.
// NOTE(yangke): another way to do this is to first load a list of the tensor
// slices needed and then just selectively read some of the meta data. That
// might optimize the loading but makes the logic a bit more complicated. We
// might want to revisit that.
// TODO(yangke): consider moving to TensorProto.
class TensorSliceReader {
 public:
  // Abstract interface for reading data out of a tensor slice checkpoint file
  class Table {
   public:
    virtual ~Table();
    virtual bool Get(const string& key, string* value) = 0;
  };
  typedef std::function<Status(const string&, Table**)> OpenTableFunction;

  static const int kLoadAllShards = -1;
  TensorSliceReader(const string& filepattern, OpenTableFunction open_function);
  TensorSliceReader(const string& filepattern, OpenTableFunction open_function,
                    int preferred_shard);
  virtual ~TensorSliceReader();

  // Get the filename this reader is attached to.
  const string& filepattern() const { return filepattern_; }

  // Get the number of files matched.
  int num_files() const { return sss_.size(); }

  // Get the status of the reader.
  const Status status() const { return status_; }

  // Checks if the reader contains any slice of a tensor. In case the reader
  // does contain the tensor, if "shape" is not nullptr, fill "shape" with the
  // shape of the tensor; if "type" is not nullptr, fill "type" with the type
  // of the tensor.
  bool HasTensor(const string& name, TensorShape* shape, DataType* type) const;

  // Checks if the reader contains all the data about a tensor slice, and if
  // yes, copies the data of the slice to "data". The caller needs to make sure
  // that "data" points to a buffer that holds enough data.
  // This is a slow function since it needs to read sstables.
  template <typename T>
  bool CopySliceData(const string& name, const TensorSlice& slice,
                     T* data) const;

  // Get the tensors.
  const std::unordered_map<string, TensorSliceSet*>& Tensors() const {
    return tensors_;
  }

 private:
  friend class TensorSliceWriteTestHelper;

  void LoadShard(int shard) const;
  void LoadAllShards() const;
  void RegisterTensorSlice(const string& name, const TensorShape& shape,
                           DataType type, const string& tag,
                           const TensorSlice& slice) const;

  const TensorSliceSet* FindTensorSlice(
      const string& name, const TensorSlice& slice,
      std::vector<std::pair<TensorSlice, string>>* details) const;

  const string filepattern_;
  const OpenTableFunction open_function_;
  std::vector<string> fnames_;
  std::unordered_map<string, int> fname_to_index_;

  // Guards the attributes below.
  mutable mutex mu_;
  mutable bool all_shards_loaded_ = false;
  mutable std::vector<std::unique_ptr<Table>> sss_;
  mutable std::unordered_map<string, TensorSliceSet*> tensors_;
  mutable Status status_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorSliceReader);
};

Status OpenTableTensorSliceReader(const string& fname,
                                  TensorSliceReader::Table** table);

template <typename T>
bool TensorSliceReader::CopySliceData(const string& name,
                                      const TensorSlice& slice, T* data) const {
  std::vector<std::pair<TensorSlice, string>> details;
  const TensorSliceSet* tss;
  {
    mutex_lock l(mu_);
    tss = FindTensorSlice(name, slice, &details);
    if (!tss && !all_shards_loaded_) {
      VLOG(1) << "Did not find slice in preferred shard, loading all shards."
              << name << ": " << slice.DebugString();
      LoadAllShards();
      tss = FindTensorSlice(name, slice, &details);
    }
    if (!tss) {
      // No such tensor
      return false;
    }
  }
  // We have the data -- copy it over.
  string value;
  for (const auto& x : details) {
    const TensorSlice& slice_s = x.first;
    const string& fname = x.second;
    int idx = gtl::FindWithDefault(fname_to_index_, fname, -1);
    CHECK_GE(idx, 0) << "Failed to find the index for filename " << fname;
    // We read a record in the corresponding sstable
    const string key = EncodeTensorNameSlice(name, slice_s);
    CHECK(sss_[idx]->Get(key, &value))
        << "Failed to seek to the record for tensor " << name << ", slice "
        << slice_s.DebugString() << ": computed key = " << key;
    SavedTensorSlices sts;
    CHECK(ParseProtoUnlimited(&sts, value))
        << "Failed to parse the record for tensor " << name << ", slice "
        << slice_s.DebugString() << ": computed key = " << key;
    CopyDataFromTensorSliceToTensorSlice(
        tss->shape(), slice_s, slice,
        checkpoint::TensorProtoData<T>(sts.data().data()), data);
  }
  return true;
}

}  // namespace checkpoint

}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_TENSOR_SLICE_READER_H_
