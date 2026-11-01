// n_widetile{8,16,32} — find where ILP stops paying and spilling starts on AVX2.
//
// Default n_planar uses an 8-frame tile (one YMM per channel). Wider tiles buy
// more independent FMA chains and amortise the weight broadcast; past the
// register file they spill. Measured on the i5-9500T, not inherited from ARM.

#if defined(NB_ENABLE_X64_LAB)

  #include "x64_planar_kernel.h"

namespace x64lab
{
namespace
{

struct Options8 : NanoPlanarOptions
{
  static constexpr int kTile = 8;
};

struct Options16 : NanoPlanarOptions
{
  static constexpr int kTile = 16;
};

struct Options32 : NanoPlanarOptions
{
  static constexpr int kTile = 32;
};

} // namespace

std::unique_ptr<nam::DSP> make_nano_widetile8(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options8>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_widetile16(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options16>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_widetile32(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options32>>(weights, sampleRate);
}

} // namespace x64lab

#endif // NB_ENABLE_X64_LAB
