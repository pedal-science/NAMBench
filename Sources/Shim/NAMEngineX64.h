// Umbrella header for the x86_64 AVX2 kernel lab variant framework.
//
// Built from the same sdatkinson/NeuralAmpModelerCore tree as the upstream
// variant, through the same target template, so the only difference between the
// two builds is NB_ENABLE_X64_LAB and the extra sources under Sources/X64Engines.

#pragma once

#include "nam_bench_shim.h"

NB_DECLARE_VARIANT(nb_x64)
NB_DECLARE_KERNEL_LAB(nb_x64)
