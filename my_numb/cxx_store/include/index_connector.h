#ifndef INDEX_CONNECTOR_H_
#define INDEX_CONNECTOR_H_

#include <vector>

#include "open_iterator.h"

using rest::OpenIterator;
using std::pair;

void border_init(const std::vector<int>& borders);

void rest_add(int label, uint64_t tl_points_pack, uint64_t sl_id_pack, uint64_t s_beg);

pair<OpenIterator<int64_t>, OpenIterator<int64_t>> rest_query(int label, const std::vector<int>& region, int duration_lower, const std::vector<int>& time_interval, int mode);

/********** the rest of code is only for testing ***********/
std::vector<int> get_border(int label, int type);
#endif // INDEX_CONNECTOR_H_