/* The meta of a unit library: constant data and one exported function. Generated. */
#include "eagle/door/eg_door.h"
#include "eagle/door/eg_unit_meta.h"
#ifdef __cplusplus
#define M(member, ...) __VA_ARGS__
#else
#define M(member, ...) .member = __VA_ARGS__
#endif
static const struct eg_unit_slot u0_s0 = {M(struct_size, sizeof(struct eg_unit_slot)), M(flags, 0u), M(role, EG_ROLE_MUTABLE), M(dtype, EG_DTYPE_REAL), M(rank, 0), M(like_slot, -1), M(n_gather, 0), M(reserved0, 0), M(shape, NULL), M(gather, NULL), M(name, "y")};
static const struct eg_unit_slot u0_s1 = {M(struct_size, sizeof(struct eg_unit_slot)), M(flags, 0u), M(role, EG_ROLE_PER_SAMPLE), M(dtype, EG_DTYPE_REAL), M(rank, 0), M(like_slot, -1), M(n_gather, 0), M(reserved0, 0), M(shape, NULL), M(gather, NULL), M(name, "a")};
static const struct eg_unit_slot u0_s2 = {M(struct_size, sizeof(struct eg_unit_slot)), M(flags, 0u), M(role, EG_ROLE_PER_SAMPLE), M(dtype, EG_DTYPE_REAL), M(rank, 0), M(like_slot, -1), M(n_gather, 0), M(reserved0, 0), M(shape, NULL), M(gather, NULL), M(name, "x")};
static const struct eg_unit_slot u0_s3 = {M(struct_size, sizeof(struct eg_unit_slot)), M(flags, 0u), M(role, EG_ROLE_UNIFORM), M(dtype, EG_DTYPE_REAL), M(rank, 0), M(like_slot, -1), M(n_gather, 0), M(reserved0, 0), M(shape, NULL), M(gather, NULL), M(name, "p")};
static const struct eg_unit_slot* const u0_slots[] = {&u0_s0, &u0_s1, &u0_s2, &u0_s3};
static const struct eg_unit_body u0_body = {M(struct_size, sizeof(struct eg_unit_body)), M(access, EG_ACCESS_SAMPLE_LOCAL), M(n_slots, 4), M(n_companions, 0), M(n_cells, 0), M(n_keyed, 0), M(n_team_planes, 0), M(n_dispatches, 0), M(cell_row, 0), M(scatter_chunks, 0), M(slots, u0_slots), M(companions, NULL), M(cells, NULL), M(keyed, NULL), M(team_planes, NULL), M(dispatches, NULL)};
static const char* const u0_lane_labels[] = {NULL};
static const int64_t u0_lane_extents[] = {-1};
static const struct eg_unit_launch u0_launch = {M(struct_size, sizeof(struct eg_unit_launch)), M(n_lane_axes, 1), M(block, 0), M(active_map_slot, -1), M(active_count_slot, -1), M(index_map_slot, -1), M(index_count_slot, -1), M(n_plane_axes, 0), M(group, 1), M(lane_labels, u0_lane_labels), M(lane_extents, u0_lane_extents), M(plane_axes, NULL), M(host_box, NULL), M(device_box, NULL)};
static const struct eg_unit_meta u0_meta = {M(struct_size, sizeof(struct eg_unit_meta)), M(flags, 0u), M(name, "ss_bcast"), M(digest, {0x49, 0xa1, 0x2f, 0x72, 0xdf, 0xd7, 0x52, 0x0a, 0xa7, 0x6e, 0xd0, 0x7e, 0x9c, 0x47, 0x65, 0xed, 0x5f, 0x4c, 0x5a, 0x6d, 0x4d, 0x2b, 0x29, 0xb5, 0xad, 0x19, 0xeb, 0x91, 0x76, 0x7b, 0xef, 0xec}), M(pattern, EG_PATTERN_PURE), M(scalar_type, EG_SCALAR_FLOAT64), M(host_entry, "ss_bcast_host"), M(device_image, NULL), M(device_function, NULL), M(device_image_bytes, 0), M(device_image_format, EG_IMAGE_NONE), M(n_stages, 0), M(stages, NULL), M(body, &u0_body), M(launch, &u0_launch), M(finish, NULL), M(device_image_size, NULL)};
static const struct eg_unit_meta* const lib_units[] = {&u0_meta};
static const struct eg_library_meta lib_meta = {M(struct_size, sizeof(struct eg_library_meta)), M(schema_major, EG_META_SCHEMA_MAJOR), M(schema_minor, 0u), M(n_units, 1u), M(units, lib_units)};
__attribute__((visibility("default"))) const struct eg_library_meta* eg_unit_library_meta(void) { return &lib_meta; }
