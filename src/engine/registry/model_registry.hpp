#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "../../model/model.hpp"

namespace llaisys::model {

using ModelFactory = std::function<model_t(llaisysDeviceType_t device, int device_id)>;

class ModelRegistry {
public:
    static void register_model(const std::string &arch, ModelFactory factory);
    static model_t create(const std::string &arch, llaisysDeviceType_t device, int device_id);
};

} // namespace llaisys::model
