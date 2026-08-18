#pragma once

#include "../model.hpp"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace llaisys::model {

using ModelFactory = std::function<std::shared_ptr<Model>(llaisysDeviceType_t device, int *device_ids, int ndevice)>;

class ModelRegistry {
public:
    static void register_model(const std::string &arch, ModelFactory factory);
    static std::shared_ptr<Model> create(const std::string &arch, llaisysDeviceType_t device, int *device_ids,
                                         int ndevice);
};

} // namespace llaisys::model
