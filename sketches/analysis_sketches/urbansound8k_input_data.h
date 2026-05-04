#pragma once

// Placeholder UrbanSound8K log-mel input profile.
// Regenerate from a local cache with:
//   python analysis_scripts/hil_noise_analysis/urbansound8k_input_profile.py --split calibration --export-header sketches/analysis_sketches/urbansound8k_input_data.h

#include <stdint.h>

static const int kInputWindowSize = 201;
static const int kInputChannels = 64;
static const int kRealWindowCount = 1;

static const float kChannelMeans[kInputChannels] = {0.0f};
static const float kChannelStds[kInputChannels] = {
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
  1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f
};
static const float kChannelMin[kInputChannels] = {0.0f};
static const float kChannelMax[kInputChannels] = {0.0f};
static const uint8_t kChannelIsBinary[kInputChannels] = {0};

static const float kRealWindows[
    kRealWindowCount * kInputWindowSize * kInputChannels
] = {0.0f};
