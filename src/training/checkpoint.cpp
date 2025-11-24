// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "checkpoint.h"

#include <filesystem>

#include <nlohmann/json.hpp>
#include <fmt/core.h>

#include "dataloader.h"
#include "model.h"
#include "utilities/comm.h"
#include "utilities/safetensors.h"
#include "utilities/tensor.h"

std::string get_checkpoint_path(std::string checkpoint_directory, int step) {
    checkpoint_directory += fmt::format("/step_{:08}", step);
    return checkpoint_directory;
}

std::string save_checkpoint(std::string target, int step, IModel& model, const DataLoader* loader, NCCLCommunicator& comm) {
    comm.barrier();

    nlohmann::json meta_data;

    target = get_checkpoint_path(std::move(target), step);
    std::filesystem::create_directories(target);

    // weights
    // TODO don't duplicate weights if they are unsharded
    write_safetensors(target + fmt::format("/weights.shard_{:03}_of_{:03}.safetensors", comm.rank(), comm.world_size()), model.weights());

    // sharded optimizer state
    write_safetensors(target + fmt::format("/adam.m.shard_{:03}_of_{:03}.safetensors", comm.rank(), comm.world_size()), model.opt_momentum());
    write_safetensors(target + fmt::format("/adam.v.shard_{:03}_of_{:03}.safetensors", comm.rank(), comm.world_size()), model.opt_variance());

    bool has_scales = false;
    model.opt_momentum_scales().iterate_tensors([&has_scales](const std::string& name, const TensorShard& tensor){
        if(tensor.Data != nullptr) {
            has_scales = true;
        }
    });
    if(has_scales) {
        write_safetensors(target + fmt::format("/adam.m.scales.shard_{:03}_of_{:03}.safetensors", comm.rank(), comm.world_size()), model.opt_momentum_scales());
    }

    comm.barrier();  // only write checkpoint.json once we know all the shard files are saved

    if (comm.rank() == 0) {
        if(loader) {
            meta_data["data-loader"] = nlohmann::json::object({
                  {"seed",        loader->seed()},
                  {"chunk_index", loader->chunk_index()},
                  {"file_index",  loader->file_index()},
                  {"epoch",       loader->epoch()}
            });
        }

        meta_data["run"] = nlohmann::json::object({
            {"step", step},
            {"rng", model.rng_state()},
        });

        meta_data["distributed"] = nlohmann::json::object({
            {"world", comm.world_size()},
        });

        std::ofstream file(target + "/checkpoint.json");
        file << std::setw(2) << meta_data;
    }

    return target;
}

void load_checkpoint(std::string source, int step, IModel& model, DataLoader* loader, NCCLCommunicator& comm) {
    comm.barrier();
    source = get_checkpoint_path(std::move(source), step);
    if(!std::filesystem::exists(source)) {
        throw std::runtime_error("Checkpoint not found: " + source);
    }

    std::ifstream file(source + "/checkpoint.json");
    if(!file.is_open()) {
        throw std::runtime_error(fmt::format("could not open config file {}", source + "/checkpoint.json"));
    }

    nlohmann::json meta_data = nlohmann::json::parse(file);
    if(int ws = meta_data["distributed"]["world"].get<int>(); ws != comm.world_size()) {
        throw std::runtime_error(
            fmt::format("Loading checkpoints with different world size is not supported: Current world size: {}, checkpoint world size: {}",
                        comm.world_size(), ws));
    }

    model.set_rng_state(meta_data["run"]["rng"].get<std::vector<std::byte>>());

    if (loader) {
        const auto& dl = meta_data["data-loader"];
        loader->set_state(dl["seed"].get<std::uint64_t>(), dl["epoch"].get<int>(), dl["file_index"].get<int>(), dl["chunk_index"].get<int>());
    }

    // weights
    load_safetensors(source + fmt::format("/weights.shard_{:03}_of_{:03}.safetensors", comm.rank(), comm.world_size()), model.weights(), false);

    // load optimizer shards
    load_safetensors(source + fmt::format("/adam.m.shard_{:03}_of_{:03}.safetensors", comm.rank(), comm.world_size()), model.opt_momentum(), false);
    load_safetensors(source + fmt::format("/adam.v.shard_{:03}_of_{:03}.safetensors", comm.rank(), comm.world_size()), model.opt_variance(), false);

    bool has_scales = false;
    model.opt_momentum_scales().iterate_tensors([&has_scales](const std::string& name, const TensorShard& tensor){
        if(tensor.Data != nullptr) {
            has_scales = true;
        }
    });
    if(has_scales) {
        load_safetensors(source + fmt::format("/adam.m.scales.shard_{:03}_of_{:03}.safetensors", comm.rank(), comm.world_size()), model.opt_momentum_scales(), false);
    }

    model.on_restore_checkpoint(comm);
}

int get_checkpoint_world_size(std::string checkpoint_directory, int step) {
    std::string path = get_checkpoint_path(checkpoint_directory, step);
    if(!std::filesystem::exists(path)) {
        throw std::runtime_error("Checkpoint not found: " + path);
    }

    std::ifstream file(path + "/checkpoint.json");
    if(!file.is_open()) {
        throw std::runtime_error(fmt::format("could not open config file {}", path + "/checkpoint.json"));
    }
    nlohmann::json meta_data = nlohmann::json::parse(file);
    return meta_data["distributed"]["world"].get<int>();
}

std::vector<int> get_all_checkpoints(const std::string& checkpoint_directory) {
    std::filesystem::path path(checkpoint_directory);
    if (!exists(path)) {
        return {};
    }
    std::filesystem::directory_iterator end_iter;
    std::vector<int> checkpoints;
    for(auto it = std::filesystem::directory_iterator(path); it != end_iter; ++it) {
        if(std::filesystem::is_directory(*it)) {
            std::string name = it->path().filename().string();
            if(name.starts_with("step_")) {
                int step = std::stoi(name.substr(5));
                if(step > 0) {
                    checkpoints.push_back(step);
                }
            }
        }
    }
    return checkpoints;
}

int find_latest_checkpoint(const std::string& checkpoint_directory) {
    auto checkpoints = get_all_checkpoints(checkpoint_directory);
    return checkpoints.empty() ? -1 : *std::max_element(checkpoints.begin(), checkpoints.end());
}


std::vector<std::string> clean_old_checkpoints(const std::string& checkpoint_directory, int n_to_keep, int major_every) {
    auto checkpoints = get_all_checkpoints(checkpoint_directory);
    if(checkpoints.size() <= n_to_keep) {
        return {};
    }

    std::vector<std::string> removed;
    // leave major checkpoints untouched
    if(major_every > 0) {
        std::erase_if(checkpoints, [&](int step) { return step % major_every == 0; });
    }
    std::sort(checkpoints.begin(), checkpoints.end());
    for(int i = 0; i < checkpoints.size() - n_to_keep; ++i) {
        std::string path = get_checkpoint_path(checkpoint_directory, checkpoints[i]);
        removed.push_back(path);
        std::filesystem::remove_all(path);
    }

    return removed;
}
