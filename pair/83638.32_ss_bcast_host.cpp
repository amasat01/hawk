// HAWK-emitted host translation unit (scalar mode: float64).
// Body text is the ONE renderer's; this file is the entry wrapper.

#include <aether/typedefs.h>
#include <aether/device.h>
#include <aether/expr/nodes/Arithmetic.h>
#include <aether/expr/nodes/Constant.h>
#include <aether/expr/nodes/Elementwise.h>
#include <aether/expr/nodes/Geometric.h>
#include <aether/expr/nodes/Inverse.h>
#include <aether/expr/nodes/Product.h>
#include <aether/expr/nodes/Quaternion.h>
#include <aether/expr/nodes/Structural.h>
#include <aether/expr/Reduce.h>
#include <aether/math/math.h>

#include "plugin/gref_layout.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

using Real = double;
using Int = long long;

static inline int hawk_clamp_count(long long n)
{
    return n < 2147483647LL ? static_cast<int>(n) : 2147483647;
}

namespace hawk_abi {

template <class T, std::size_t N>
static inline aether::View<T, aether::extents<N, aether::dyn>, aether::layout_right>
vec_view(const eagle::plugin::GRefMirror& m)
{
    // A GRef plane is SoA and C-contiguous by eagle's packing rule (marshal.py):
    // unit sample stride, component pitch = m.compStride_ (the full plane's
    // sample count even for a partition slice). layout_right over that ONE
    // runtime pitch lets aether fold the per-component offsets.
    using Ext = aether::extents<N, aether::dyn>;
    using Map = aether::layout_right::mapping<Ext>;
    return aether::View<T, Ext, aether::layout_right>(
        reinterpret_cast<T*>(m.data_),
        Map(Ext(static_cast<std::size_t>(m.compStride_))),
        aether::Device(static_cast<DLDeviceType>(m.deviceType_), m.deviceId_));
}

template <class T, std::size_t R, std::size_t C>
static inline aether::View<T, aether::extents<R, C, aether::dyn>, aether::layout_right>
mat_view(const eagle::plugin::GRefMirror& m)
{
    using Ext = aether::extents<R, C, aether::dyn>;
    using Map = aether::layout_right::mapping<Ext>;
    return aether::View<T, Ext, aether::layout_right>(
        reinterpret_cast<T*>(m.data_),
        Map(Ext(static_cast<std::size_t>(m.compStride_))),
        aether::Device(static_cast<DLDeviceType>(m.deviceType_), m.deviceId_));
}

template <class T>
static inline aether::View<T, aether::extents<aether::dyn>, aether::layout_right>
scalar_view(const eagle::plugin::ScalarHandle& m)
{
    using Ext = aether::extents<aether::dyn>;
    using Map = aether::layout_right::mapping<Ext>;
    return aether::View<T, Ext, aether::layout_right>(
        reinterpret_cast<T*>(m.data), Map(Ext(static_cast<std::size_t>(m.samples))),
        aether::Device(static_cast<DLDeviceType>(m.deviceType), m.deviceId));
}

template <class ViewT, class T>
static inline void accum_add(ViewT& target, const aether::SampleIndex& at, T term)
{
    // The scattered commit (hawk/ext): ONE body string spelled per
    // target here -- the device's own atomic add, or an atomic_ref under the
    // OpenMP host team -- because a scattered read-modify-write has no correct
    // use under any parallel launch. One subscript on the target.
    auto& slot = target(at.global());
    using E = std::remove_reference_t<decltype(slot)>;
    static_assert(std::is_floating_point_v<E>,
                  "hawk_abi::accum_add: a scattered accumulate commits through the "
                  "target's atomic add, and only a floating accumulate plane has one");
    std::atomic_ref<E>(slot).fetch_add(static_cast<E>(term), std::memory_order_relaxed);
}

}  // namespace hawk_abi



// The self-check exports, DERIVED from this build.
extern "C" const char eagle_abi_tag[] = "aether-abi/2";
extern "C" const std::uint64_t eagle_layout_sizes[5] = {
    sizeof(eagle::plugin::GRefMirror),
    sizeof(eagle::plugin::ScalarHandle),
    sizeof(eagle::plugin::IntHandle),
    sizeof(EAGLE_ABI_INDEX_T),
    sizeof(eagle::plugin::PartitionTriple),
};


extern "C" void ss_bcast_host(void* const* params,
    std::int64_t base,
    std::int64_t count,
    std::int64_t nSamples)
{
    aether::View<Real, aether::extents<aether::dyn>, aether::layout_right> mut_y =
        hawk_abi::scalar_view<Real>(*static_cast<const eagle::plugin::ScalarHandle*>(params[0]));
    (void)mut_y;
    const aether::View<Real, aether::extents<aether::dyn>, aether::layout_right> psc_a =
        hawk_abi::scalar_view<Real>(*static_cast<const eagle::plugin::ScalarHandle*>(params[1]));
    (void)psc_a;
    const aether::View<Real, aether::extents<aether::dyn>, aether::layout_right> psc_x =
        hawk_abi::scalar_view<Real>(*static_cast<const eagle::plugin::ScalarHandle*>(params[2]));
    (void)psc_x;
    const Real uni_p =
        *static_cast<const Real*>(params[3]);
    (void)uni_p;
    const aether::offset_t hawk_lo =
        static_cast<aether::offset_t>(base);
    const aether::offset_t hawk_hi =
        static_cast<aether::offset_t>(base + count);
#if defined(__clang__)
#pragma clang loop vectorize(assume_safety)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
    for (aether::offset_t hawk_j = hawk_lo; hawk_j < hawk_hi; ++hawk_j) {
        const std::int64_t hawk_i = static_cast<std::int64_t>(hawk_j);
        const aether::SampleIndex i =
            aether::SampleIndex::make(static_cast<std::size_t>(hawk_j));
        (void)hawk_i;
        (void)nSamples;
        mut_y[i] = ((psc_x[i].eval() * psc_a[i].eval()) + uni_p);
    }
}
