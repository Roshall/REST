//
// Created by lg on 24-4-26.
//

#ifndef INDEX_REST_QUERY_H_
#define INDEX_REST_QUERY_H_

#include "iterator.h"

#include <memory>
#include <vector>

namespace rest {
using std::unique_ptr;
using std::vector;
using std::pair;

//struct SequenceTrajectory {
//  int64_t begin;
//};
using record_type = int64_t;

class OpenIterator {
  mutable unique_ptr<Iterator> it_;
 public:
  OpenIterator() = default;
  explicit OpenIterator(Iterator *it) : it_(it) {}
  explicit OpenIterator(unique_ptr<Iterator> &&it) : it_(std::move(it)) {}
  OpenIterator(OpenIterator const &other) : it_(std::move(other.it_)) {}
  OpenIterator &operator=(const rest::OpenIterator &other) noexcept {
    it_ = std::move(other.it_);
    return *this;
  }
  OpenIterator(OpenIterator &other) : it_(other.it_.release()) {}
  inline int HasNext() { return it_->HasNext(); }
  inline uint64_t Next() { return *static_cast<record_type *>(it_->Next()); }
};

class RestIndex {
 private:
  class Impl;
  unique_ptr<Impl> pimpl_{};
 public:
  using comb_key_type = pair<pair<int, int>, pair<int, pair<int, int>>>;
  RestIndex();
  ~RestIndex();
  RestIndex(RestIndex && other) noexcept;
  void Build(const vector<int> &region_border_stride,
             const vector<int> &tempo_border);
  void Add(const comb_key_type &key, int64_t seg);
  pair<OpenIterator, OpenIterator> Query(const vector<int> &spat_region,
                                         int duration_lower,
                                         const pair<int, int> &time_interval,
                                         int pseudo_mode);
};
}
#endif //INDEX_REST_QUERY_H_
