#ifndef INDEX_CONNECTOR_H_
#define INDEX_CONNECTOR_H_

#include <vector>


void border_init(const std::vector<int>& borders);

// tl_points_pack format: (trajecotry lifelong, points_y, points_x) |----4 bytes-----|----2 bytes-----|----2 bytes-----|
// sl_id_pack format: (segment lifelong, trajectory id) |----4 bytes-----|----4 bytes-----|
void rest_add(int label, uint64_t tl_points_pack, uint64_t sl_id_pack, int64_t s_beg);

// region format: (x_min, x_max, y_min, y_max), time_interval format: (t_min, t_max)
void rest_query(int label, const std::vector<int>& region, int duration_lower, const std::vector<int>& time_interval, int mode);

int iter_has_next(int sure);
int64_t iter_next(int sure);

/********** the rest of code is only for testing ***********/
std::vector<int> get_border(int label, int type);
#endif // INDEX_CONNECTOR_H_