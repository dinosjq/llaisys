#include "model_registry.hpp"

#include <mutex>
#include <unordered_map>

namespace llaisys::model {

namespace {

std::mutex g_mu;
std::unordered_map<std::string, ModelFactory> g_factories;

} // namespace

void ModelRegistry::register_model(const std::string &arch, ModelFactory factory) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_factories[arch] = std::move(factory);
}

model_t ModelRegistry::create(const std::string &arch, llaisysDeviceType_t device, int device_id) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_factories.find(arch);
    if (it == g_factories.end()) {
        return nullptr;
    }
    return it->second(device, device_id);
}

} // namespace llaisys::model
