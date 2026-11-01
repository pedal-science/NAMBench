// AVX2 / FMA helpers for the x86_64 kernel lab.
//
// Mirrors a32_neon_compat.h's role: this is the only file that knows the
// instruction set. The planar kernel is written against VecWidth<LANES> and
// compiles for 8-wide (YMM) or 4-wide (XMM) tiles.
//
// Weight delivery is always a broadcast from memory (_mm256_broadcast_ss /
// _mm_broadcast_ss). There is no by-element FMA on AVX2 — that is AVX-512 —
// so the ARMv7 lesson applies here too: do not hold a weight vector just to
// pick a lane out of it.
//
// round_now blocks -ffp-contract=fast from folding the Eigen C=8 mixin's
// product-then-add back into a single FMA. Same reason as on ARM.

#pragma once

#if defined(NB_ENABLE_X64_LAB)

  #include <immintrin.h>

  #if !defined(__AVX2__) || !defined(__FMA__)
    #error "the X64 lab needs AVX2 and FMA: build with -mavx2 -mfma"
  #endif

  #define NB_X64_INLINE inline __attribute__((always_inline))

namespace x64lab
{
namespace compat
{

NB_X64_INLINE __m256 round_now(__m256 v)
{
  __asm__("" : "+x"(v));
  return v;
}

NB_X64_INLINE __m128 round_now(__m128 v)
{
  __asm__("" : "+x"(v));
  return v;
}

NB_X64_INLINE float round_now(float v)
{
  __asm__("" : "+x"(v));
  return v;
}

NB_X64_INLINE __m256 fmaq_bcast(__m256 acc, __m256 s, const float* w)
{
  return _mm256_fmadd_ps(s, _mm256_broadcast_ss(w), acc);
}

NB_X64_INLINE __m128 fmaq_bcast(__m128 acc, __m128 s, const float* w)
{
  return _mm_fmadd_ps(s, _mm_broadcast_ss(w), acc);
}

/// 8-lane YMM width — the natural AVX2 packet, and Eigen's AVX packet size.
struct Y8
{
  using type = __m256;
  static constexpr int kLanes = 8;

  static NB_X64_INLINE type dup(float v) { return _mm256_set1_ps(v); }
  static NB_X64_INLINE type load(const float* p) { return _mm256_loadu_ps(p); }
  static NB_X64_INLINE void store(float* p, type v) { _mm256_storeu_ps(p, v); }
  static NB_X64_INLINE type add(type a, type b) { return _mm256_add_ps(a, b); }
  static NB_X64_INLINE type mul(type a, type b) { return _mm256_mul_ps(a, b); }
  static NB_X64_INLINE type fma_bcast(type acc, type s, const float* w)
  {
    return _mm256_fmadd_ps(s, _mm256_broadcast_ss(w), acc);
  }
  static NB_X64_INLINE type mul_bcast(type s, const float* w)
  {
    return _mm256_mul_ps(s, _mm256_broadcast_ss(w));
  }
  static NB_X64_INLINE type leaky(type v, type slope)
  {
    const type zero = _mm256_setzero_ps();
    const type neg = _mm256_mul_ps(v, slope);
    return _mm256_blendv_ps(v, neg, _mm256_cmp_ps(v, zero, _CMP_LT_OQ));
  }
  static NB_X64_INLINE type round_now(type v) { return compat::round_now(v); }
  static NB_X64_INLINE type scale(type v, float s) { return _mm256_mul_ps(v, _mm256_set1_ps(s)); }
};

/// 4-lane XMM width — for tile 4 and for the scalar-ish tail of a wider tile.
struct X4
{
  using type = __m128;
  static constexpr int kLanes = 4;

  static NB_X64_INLINE type dup(float v) { return _mm_set1_ps(v); }
  static NB_X64_INLINE type load(const float* p) { return _mm_loadu_ps(p); }
  static NB_X64_INLINE void store(float* p, type v) { _mm_storeu_ps(p, v); }
  static NB_X64_INLINE type add(type a, type b) { return _mm_add_ps(a, b); }
  static NB_X64_INLINE type mul(type a, type b) { return _mm_mul_ps(a, b); }
  static NB_X64_INLINE type fma_bcast(type acc, type s, const float* w)
  {
    return _mm_fmadd_ps(s, _mm_broadcast_ss(w), acc);
  }
  static NB_X64_INLINE type mul_bcast(type s, const float* w)
  {
    return _mm_mul_ps(s, _mm_broadcast_ss(w));
  }
  static NB_X64_INLINE type leaky(type v, type slope)
  {
    const type zero = _mm_setzero_ps();
    const type neg = _mm_mul_ps(v, slope);
    return _mm_blendv_ps(v, neg, _mm_cmplt_ps(v, zero));
  }
  static NB_X64_INLINE type round_now(type v) { return compat::round_now(v); }
  static NB_X64_INLINE type scale(type v, float s) { return _mm_mul_ps(v, _mm_set1_ps(s)); }
};

template <int LANES>
struct VecWidth;
template <>
struct VecWidth<8>
{
  using type = Y8;
};
template <>
struct VecWidth<4>
{
  using type = X4;
};

} // namespace compat
} // namespace x64lab

#endif // NB_ENABLE_X64_LAB
