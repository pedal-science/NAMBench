// Kernel 7: n_skiplast — stop doing two pieces of work nothing reads.
//
// The last layer's layer1x1 computes a residual that no later layer consumes:
// the head reads head_sum, not layer_in. That is C × C FMAs plus a bias and a
// store per frame, thrown away.
//
// And head_sum is memset to zero every block only so the first layer can
// accumulate into it. If the first layer *stores* instead, the memset is
// unnecessary — one pass over C × N floats per block, gone.
//
// Neither changes any arithmetic that survives to the output, so the parity
// claim is untouched. Free on AArch64; free is worth more on a slower part.

#if defined(NB_ENABLE_X64_LAB)

  #include "x64_planar_kernel.h"

namespace x64lab
{
namespace
{

struct Options : NanoPlanarOptions
{
  static constexpr bool kSkipLastL1x1 = true;
};

} // namespace

std::unique_ptr<nam::DSP> make_nano_skiplast(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options>>(weights, sampleRate);
}

} // namespace x64lab

#endif // NB_ENABLE_X64_LAB
