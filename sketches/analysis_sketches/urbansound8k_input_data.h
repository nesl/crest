// Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Placeholder UrbanSound8K log-mel input profile used by firmware smoke paths.
// Replace with a real cache-derived profile before representative audio HIL.

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
