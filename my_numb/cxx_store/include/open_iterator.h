//
// Created by lg on 24-12-26.
//

#ifndef INDEX_OPEN_ITERATOR_H_
#define INDEX_OPEN_ITERATOR_H_
#include <memory>

#include "iterator.h"

namespace rest {
using std::unique_ptr;

template<typename record_type>
class OpenIterator {
  mutable unique_ptr<Iterator> it_;
 public:
  OpenIterator() = default;
  explicit OpenIterator(Iterator *it) : it_(it) {}
  explicit OpenIterator(unique_ptr<Iterator> &&it) : it_(std::move(it)) {}
  OpenIterator(OpenIterator const &other) : it_(std::move(other.it_)) {}
  OpenIterator &operator=(const OpenIterator &other) noexcept {
    it_ = std::move(other.it_);
    return *this;
  }
  OpenIterator(OpenIterator &other) : it_(other.it_.release()) {}
  inline int HasNext() { return it_->HasNext(); }
  inline uint64_t Next() { return *static_cast<record_type *>(it_->Next()); }
};
}
#endif //INDEX_OPEN_ITERATOR_H_
