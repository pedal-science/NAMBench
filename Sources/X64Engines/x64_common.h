// Shared scaffolding for the x86_64 (AVX2+FMA) kernel lab.
//
// Drop-in replacements for a2_fast's two branches, built from the same
// vendor/upstream tree. One lab for both A2 submodels — same shape as the A32
// lab, same reason: one reference, one register-file problem (16 YMMs), one
// weight-delivery story (broadcast; no by-element FMA on AVX2).

#pragma once

#if defined(NB_ENABLE_X64_LAB)

  #include <array>
  #include <memory>
  #include <vector>

  #include "NAM/dsp.h"
  #include "json.hpp"

namespace x64lab
{

constexpr int kNumLayers = 23;
constexpr int kHeadKernelSize = 16;
constexpr float kLeakySlope = 0.01f;

inline constexpr std::array<int, kNumLayers> kKernelSizes = {
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6};

inline constexpr std::array<int, kNumLayers> kDilations = {
  1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239};

template <int C>
struct Shape;
template <>
struct Shape<3>
{
  static constexpr int kWeightCount = 1871;
};
template <>
struct Shape<8>
{
  static constexpr int kWeightCount = 12146;
};

template <int C>
struct LayerWeights
{
  int kernel_size = 0;
  int dilation = 0;
  int max_lookback = 0;

  std::vector<float> conv_w;
  std::array<float, C> conv_b{};
  std::array<float, C> mixin_w{};
  std::array<float, C * C> l1x1_w{};
  std::array<float, C> l1x1_b{};
};

template <int C>
struct Weights
{
  std::array<float, C> rechannel_w{};
  std::array<LayerWeights<C>, kNumLayers> layers;
  std::array<std::array<float, C>, kHeadKernelSize> head_w{};
  float head_b = 0.0f;
  float head_scale = 1.0f;
};

template <int C>
Weights<C> parse_weights(const std::vector<float>& weights);

extern template Weights<3> parse_weights<3>(const std::vector<float>&);
extern template Weights<8> parse_weights<8>(const std::vector<float>&);

int prewarm_samples();
int next_pow2(int v);

template <int C>
class X64Model : public nam::DSP
{
public:
  X64Model(const std::vector<float>& weights, double expected_sample_rate)
  : nam::DSP(/*in_channels=*/1, /*out_channels=*/1, expected_sample_rate)
  , _w(parse_weights<C>(weights))
  , _prewarm(prewarm_samples())
  {
  }

  int GetPrewarmSamples() override { return _prewarm; }

protected:
  Weights<C> _w;
  int _prewarm = 0;
};

using Factory = std::unique_ptr<nam::DSP> (*)(const std::vector<float>&, double);

struct KernelEntry
{
  const char* name;
  Factory make;
  int channels;
  bool exact;
};

int kernel_count();
const char* kernel_name(int index);
int kernel_channels(int index);
int kernel_exact(int index);

std::unique_ptr<nam::DSP> create(int index, const nlohmann::json& config, std::vector<float> weights,
                                 double sampleRate);

// Controls
std::unique_ptr<nam::DSP> make_nano_baseline(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_baseline(const std::vector<float>&, double);

// C=3 planar family
std::unique_ptr<nam::DSP> make_nano_planar(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_widetile8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_widetile16(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_widetile32(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_ringdirect(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_skiplast(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_planar_linear(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_planar_lazy(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked16(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked8_ring(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked8_skip(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked8_linear(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked8_lazy(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked32(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked32_linear(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked32_lazy(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked32_ring(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_nano_stacked32_skip(const std::vector<float>&, double);

// C=8 planar family — tile 2 dropped (not a multiple of 4 under AVX2 lane rules)
std::unique_ptr<nam::DSP> make_std_planar4(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_planar8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_planar16(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_planar32(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_ringdirect(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_skiplast(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_stacked4(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_stacked8(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_stacked16(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_stacked8_linear(const std::vector<float>&, double);
std::unique_ptr<nam::DSP> make_std_stacked8_lazy(const std::vector<float>&, double);

} // namespace x64lab

#endif // NB_ENABLE_X64_LAB
