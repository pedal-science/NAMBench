// Nano stacked compositions for the AVX2 lab.

#if defined(NB_ENABLE_X64_LAB)

  #include "x64_planar_kernel.h"

namespace x64lab
{
namespace
{

struct Options8 : NanoPlanarOptions
{
  static constexpr int kTile = 8;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
};

struct Options16 : NanoPlanarOptions
{
  static constexpr int kTile = 16;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
};

struct Options8Ring : NanoPlanarOptions
{
  static constexpr int kTile = 8;
  static constexpr bool kRingDirect = true;
};

struct Options8Skip : NanoPlanarOptions
{
  static constexpr int kTile = 8;
  static constexpr bool kSkipLastL1x1 = true;
};

struct Options8Linear : NanoPlanarOptions
{
  static constexpr int kTile = 8;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
  static constexpr RingKind kRing = RingKind::LinearRewind;
};

struct Options8Lazy : NanoPlanarOptions
{
  static constexpr int kTile = 8;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
  static constexpr RingKind kRing = RingKind::Pow2Lazy;
};

struct Options32 : NanoPlanarOptions
{
  static constexpr int kTile = 32;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
};

struct Options32Linear : NanoPlanarOptions
{
  static constexpr int kTile = 32;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
  static constexpr RingKind kRing = RingKind::LinearRewind;
};

struct Options32Lazy : NanoPlanarOptions
{
  static constexpr int kTile = 32;
  static constexpr bool kRingDirect = true;
  static constexpr bool kSkipLastL1x1 = true;
  static constexpr RingKind kRing = RingKind::Pow2Lazy;
};

struct Options32Ring : NanoPlanarOptions
{
  static constexpr int kTile = 32;
  static constexpr bool kRingDirect = true;
};

struct Options32Skip : NanoPlanarOptions
{
  static constexpr int kTile = 32;
  static constexpr bool kSkipLastL1x1 = true;
};

} // namespace

std::unique_ptr<nam::DSP> make_nano_stacked8_linear(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options8Linear>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_stacked8_lazy(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options8Lazy>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_stacked8(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options8>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_stacked16(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options16>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_stacked8_ring(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options8Ring>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_stacked8_skip(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options8Skip>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_stacked32(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options32>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_stacked32_linear(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options32Linear>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_stacked32_lazy(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options32Lazy>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_stacked32_ring(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options32Ring>>(weights, sampleRate);
}

std::unique_ptr<nam::DSP> make_nano_stacked32_skip(const std::vector<float>& weights, double sampleRate)
{
  return std::make_unique<PlanarModel<Options32Skip>>(weights, sampleRate);
}

} // namespace x64lab

#endif // NB_ENABLE_X64_LAB
