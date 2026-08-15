#pragma once

namespace llaisys::framework {

enum class WeightRole : int {
    InEmbed,
    OutEmbed,
    OutNorm,
    AttnNorm,
    AttnQ_W,
    AttnQ_B,
    AttnK_W,
    AttnK_B,
    AttnV_W,
    AttnV_B,
    AttnO_W,
    MlpNorm,
    MlpGate_W,
    MlpUp_W,
    MlpDown_W,
};

} // namespace llaisys::framework
