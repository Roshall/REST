//
// Created by lg on 24-4-23.
//

#ifndef INDEX_ITERATOR_H_
#define INDEX_ITERATOR_H_
namespace rest{
class Iterator {
 public:
  virtual bool HasNext() = 0;
  virtual void* Next() = 0;
  virtual ~Iterator() = default;
};
}
#endif //INDEX_ITERATOR_H_

