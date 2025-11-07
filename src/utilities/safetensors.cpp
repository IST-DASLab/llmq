// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include "safetensors.h"

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "allocator.h"
#include "comm.h"
#include "cu_file.h"
#include "tensor.h"

struct sSafeTensorsHeader {
    std::uint64_t HeaderSize;
    nlohmann::json MetaData;
};

sSafeTensorsHeader read_safetensors_header(const std::string& file_name) {
    std::uint64_t header_size = -1;
    std::ifstream file(file_name, std::ios_base::binary);
    file.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
    if (!file) {
        // read error
        throw std::runtime_error("Error opening safetensors file '" + file_name + "'");
    }

    std::vector<char> header(header_size, '\0');
    file.read(header.data(), (long)header_size);
    auto parsed = nlohmann::json::parse(header.begin(), header.end());
    return {header_size, std::move(parsed)};
}

// SafeTensorEntry implementation
SafeTensorEntry::SafeTensorEntry(const std::string& name, const std::vector<long>& shape, ETensorDType dtype,
                                 std::string file_name, std::shared_ptr<cuFileRef> handle, SafeTensorsReader* reader,
                                 std::ptrdiff_t data_begin, std::ptrdiff_t data_end)
    : mName(name), mShape(shape), mDType(dtype), mFileName(std::move(file_name)), mHandle(handle), mReader(reader),
      mDataBegin(data_begin), mDataEnd(data_end) {
}

void SafeTensorEntry::read_raw(Tensor& target, std::ptrdiff_t offset,
                               std::ptrdiff_t elements, bool allow_cast) const {
    long nelem = (mDataEnd - mDataBegin) / get_dtype_size(mDType);
    if (offset < 0 || offset + elements > nelem)
        throw std::runtime_error(fmt::format("Invalid read range: offset={}, elements={}, size={}",
                                             offset, elements, nelem));

    // Check if target has enough space (in bytes)
    if (target.bytes() != elements * get_dtype_size(target.DType))
        throw std::runtime_error(fmt::format("Target tensor size mismatch for `{}`: has {} bytes, needs {} elements of {} bytes",
                                                 mName, target.bytes(), elements, get_dtype_size(target.DType)));

    std::ptrdiff_t start = mDataBegin + offset * get_dtype_size(mDType);
    std::ptrdiff_t end = start + elements * get_dtype_size(mDType);

    // Validate dtype
    if (mDType != target.DType && !allow_cast)
        throw std::runtime_error(fmt::format("DType mismatch: tensor has {}, file has {}",
                                             dtype_to_str(target.DType), dtype_to_str(mDType)));

    if (mDType == target.DType) {
        mHandle->read_bytes(target.Data, start, end);
    } else {
        // Need conversion buffer
        if (mReader->mConversionBufferSize < end - start) {
            CUDA_CHECK(cudaFree(mReader->mConversionBuffer));
            mReader->mConversionBufferSize = std::min(end - start, 256 * 1024 * 1024L);
            CUDA_CHECK(cudaMalloc((void**)&mReader->mConversionBuffer, mReader->mConversionBufferSize));
        }
        mHandle->read_and_convert(target.Data, start, end, mFileName,
                                  target.DType, mDType,
                                  mReader->mConversionBuffer,
                                  mReader->mConversionBufferSize);
    }
}

void SafeTensorEntry::read_tensor(Tensor& target, bool allow_cast) const {
    if (target.Rank != static_cast<int>(mShape.size()))
        throw std::runtime_error(fmt::format("Rank mismatch for tensor `{}`: expected {}, got {}",
                                             mName, mShape.size(), target.Rank));

    for (int i = 0; i < target.Rank; ++i)
        if (mShape[i] != target.Sizes[i])
            throw std::runtime_error(fmt::format("Shape mismatch for tensor `{}` at dim {}: expected {}, got {}",
                                                 mName, i, mShape[i], target.Sizes[i]));
    read_raw(target, 0, target.nelem(), allow_cast);
}

// SafeTensorsReader implementation
SafeTensorsReader::SafeTensorsReader(const std::string& file_name) {
    if (file_name.ends_with(".index.json"))
        parse_index_file(file_name);
    else
        parse_single_file(file_name);
}

void SafeTensorsReader::parse_single_file(const std::string& file_path) {
    auto [HeaderSize, MetaData] = read_safetensors_header(file_path);
    ptrdiff_t offset = HeaderSize + sizeof(HeaderSize);
    std::shared_ptr<cuFileRef> cu_file = std::make_shared<cuFileRef>(file_path);
    for (const auto& el : MetaData.items()) {
        const std::string& name = el.key();
        if (name == "__metadata__") {
            // TODO extract metadata?
            continue;
        }

        ETensorDType dtype = dtype_from_str(el.value()["dtype"].get<std::string_view>());
        auto shape = el.value()["shape"].get<std::vector<long>>();
        auto begin = el.value()["data_offsets"][0].get<std::ptrdiff_t>();
        auto end = el.value()["data_offsets"][1].get<std::ptrdiff_t>();

        mEntries.emplace_back(SafeTensorEntry{name, shape, dtype, file_path, cu_file, this,
                                              begin + offset, end + offset});
    }
}

void SafeTensorsReader::parse_index_file(const std::string& index_file) {
    std::ifstream file(index_file);
    auto parsed = nlohmann::json::parse(file);
    auto weight_map = parsed["weight_map"];

    std::unordered_set<std::string> processed_files;
    std::filesystem::path index_path(index_file);

    for (const auto& el : weight_map.items()) {
        auto f_name = el.value().get<std::string>();
        if (processed_files.contains(f_name))
            continue;
        processed_files.insert(f_name);

        std::filesystem::path full_path = index_path.parent_path() / f_name;
        parse_single_file(full_path.native());
    }
}

void SafeTensorsReader::load_tensors(ITensorContainer& container, bool allow_cast) const {
    std::unordered_map<std::string, Tensor> named_tensors;
    container.iterate_tensors([&named_tensors](std::string name, const Tensor& tensor) {
        named_tensors.emplace(std::move(name), tensor);
    });

    for (const auto& entry : mEntries)
        if (auto found = named_tensors.find(entry.name()); found != named_tensors.end())
            entry.read_tensor(found->second, allow_cast);
}

SafeTensorsReader::~SafeTensorsReader() {
    CUDA_CHECK(cudaFree(mConversionBuffer));
}

const SafeTensorEntry& SafeTensorsReader::find_entry(std::string_view name) const {
    for (auto& entry : mEntries)
        if (entry.name() == name)
            return entry;
    throw std::out_of_range("Entry not found");
}

void load_safetensors(const std::string& file_name, ITensorContainer& tensors, bool allow_cast) {
    SafeTensorsReader reader(file_name);
    reader.load_tensors(tensors, allow_cast);
}

SafeTensorWriter::SafeTensorWriter(std::string file_name) : mFileName(file_name) {
}

SafeTensorWriter::~SafeTensorWriter() {
    if (mFileDescriptor > 0) {
        std::string temp_name = mFileName + ".tmp";
        if (mMappedFile) {
            if (munmap(mMappedFile, mTotalSize) != 0)
                throw std::system_error(errno, std::system_category(), "Error unmapping file " + temp_name);
        }
        close(mFileDescriptor);
        unlink(temp_name.c_str());
    }
}

void SafeTensorWriter::register_tensor(const std::string& name, const TensorShard& tensor) {
    if (mMetaFinalized)
        throw std::logic_error("Cannot register tensor after metadata has been finalized");
    mRegisteredTensors.insert({name, {tensor.DType, std::vector<long>(tensor.GlobalShape.begin(), tensor.GlobalShape.begin() + tensor.Rank), 0, (long)tensor.global_nelem() * get_dtype_size(tensor.DType)}});
}

void SafeTensorWriter::prepare_metadata(NCCLCommunicator* comm) {
    nlohmann::json meta_data;
    meta_data["__metadata__"] = nlohmann::json::object({{"format", "pt"},
                                                        {"writer", "llmq"}});

    long offset = 0;
    for (auto& [name, tensor] : mRegisteredTensors) {
        meta_data[name]["dtype"] = dtype_to_str(tensor.DType);
        meta_data[name]["shape"] = tensor.Shape;
        tensor.Begin = offset;
        meta_data[name]["data_offsets"] = std::vector<long>{offset, offset + tensor.Size};
        offset += tensor.Size;
    }

    std::string header = meta_data.dump();
    std::uint64_t header_size = header.size();
    mHeaderSize = header_size + sizeof(header_size);

    std::vector<SafeTensorWriter*> peers;
    if (comm)
        peers = comm->host_gather(this);

    if (!comm || comm->rank() == 0) {
        std::string temp_name = mFileName + ".tmp";

        mFileDescriptor = open(temp_name.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR | S_IRGRP);
        if (mFileDescriptor == -1)
            throw std::system_error(errno, std::system_category(), "Error opening file '" + temp_name + "' for writing");
        mTotalSize = sizeof(header_size) + header_size + offset;
        if (ftruncate(mFileDescriptor, mTotalSize) < 0)
            throw std::system_error(errno, std::system_category(), "Error truncating file " + temp_name);

        std::byte* host_ptr = (std::byte*)mmap(nullptr, mTotalSize, PROT_WRITE,
                                               MAP_SHARED, mFileDescriptor, 0);
        if (host_ptr == MAP_FAILED)
            throw std::system_error(errno, std::system_category(), "Error memory-mapping file " + temp_name);

        // write the header
        std::memcpy(host_ptr, &header_size, sizeof(header_size));
        std::memcpy(host_ptr + sizeof(header_size), header.data(), header_size);

        for (auto& peer : peers) {
            peer->mMappedFile = host_ptr;
            peer->mMetaFinalized = true;
        }

        mMappedFile = host_ptr;
        mMetaFinalized = true;
    }

    if (comm)
        comm->barrier();
}

void SafeTensorWriter::write_tensor(const std::string& name, const TensorShard& tensor, NCCLCommunicator* comm) {
    if (!mMetaFinalized)
        throw std::logic_error("Cannot write tensor before metadata has been finalized");

    auto found = mRegisteredTensors.find(name);
    if (found == mRegisteredTensors.end())
        throw std::out_of_range("Invalid tensor " + name);

    if (found->second.Done)
        throw std::logic_error("Tensor " + name + " has already been written");

    if (!comm && tensor.NumShards > 1)
        throw std::logic_error("Cannot write tensor " + name + " with multiple shards without a communicator");

    if (comm && tensor.ShardIndex != comm->rank()) {
        // When a tensor is replicated instead of sharded, all calls with have ShardIdx == 0,
        // so only the root rank writes the tensor.
        found->second.Done = true;
        return;
    }

    long shard_begin = found->second.Begin + mHeaderSize + tensor.ShardIndex * tensor.bytes();
    std::vector<std::byte> data(tensor.bytes());
    CUDA_CHECK(cudaMemcpy(data.data(), tensor.Data, tensor.bytes(), cudaMemcpyDeviceToHost));
    std::memcpy(mMappedFile + shard_begin, data.data(), tensor.bytes());
    found->second.Done = true;
}

void SafeTensorWriter::write_raw(const std::string& name, std::ptrdiff_t offset,
                                 std::ptrdiff_t elements, const TensorShard& tensor) {
    if (!mMetaFinalized)
        throw std::logic_error("Cannot write tensor before metadata has been finalized");

    auto found = mRegisteredTensors.find(name);
    if (found == mRegisteredTensors.end())
        throw std::out_of_range("Invalid tensor " + name);

    ETensorDType dtype = tensor.DType;
    long nelem = found->second.Size / get_dtype_size(dtype);
    if (offset < 0 || offset + elements > nelem)
        throw std::logic_error(fmt::format("Invalid write range for tensor `{}`: offset={}, elements={}, size={}",
                                           name, offset, elements, nelem));

    if (found->second.DType != dtype)
        throw std::logic_error(fmt::format("DType mismatch for tensor `{}`: registered as {}, writing as {}",
                                           name, dtype_to_str(found->second.DType), dtype_to_str(dtype)));

    std::ptrdiff_t write_start = found->second.Begin + mHeaderSize + offset * get_dtype_size(dtype);
    std::ptrdiff_t write_size = elements * get_dtype_size(dtype);

    std::vector<std::byte> host_data(write_size);
    CUDA_CHECK(cudaMemcpy(host_data.data(), tensor.Data, write_size, cudaMemcpyDeviceToHost));
    std::memcpy(mMappedFile + write_start, host_data.data(), write_size);
}

void SafeTensorWriter::mark_done(const std::string& name) {
    auto found = mRegisteredTensors.find(name);
    if (found == mRegisteredTensors.end())
        throw std::out_of_range("Invalid tensor " + name);
    found->second.Done = true;
}

void SafeTensorWriter::finalize(NCCLCommunicator* comm) {
    if (comm)
        comm->barrier();

    for (auto& [name, tensor] : mRegisteredTensors)
        if (!tensor.Done)
            throw std::logic_error("Tensor " + name + " has not been written");

    if (comm)
        comm->barrier();

    if (!comm || comm->rank() == 0) {
        if (munmap(mMappedFile, mTotalSize) != 0)
            throw std::system_error(errno, std::system_category(), "Error unmapping file " + mFileName);
        mMappedFile = nullptr;
        close(mFileDescriptor);
        mFileDescriptor = -1;
        std::string temp_name = mFileName + ".tmp";
        std::filesystem::rename(temp_name, mFileName);
    }
}

void write_safetensors(const std::string& file_name, ITensorContainer& tensors) {
    SafeTensorWriter writer(file_name);
    tensors.iterate_tensors([&writer](std::string name, const Tensor& tensor) {
        writer.register_tensor(name, tensor);
    });
    writer.prepare_metadata(nullptr);
    tensors.iterate_tensors([&writer](std::string name, const Tensor& tensor) {
        writer.write_tensor(name, tensor, nullptr);
    });
    writer.finalize(nullptr);
}

std::string get_hf_hub() {
    const char* hf_home_env = std::getenv("HF_HOME");
    if (hf_home_env == nullptr) {
        const char* xdg_cache_home = std::getenv("XDG_CACHE_HOME");
        if (xdg_cache_home == nullptr)
            return std::string(std::getenv("HOME")) + "/.cache/huggingface/hub";
        else
            return std::string(xdg_cache_home) + "/huggingface/hub";
    }
    return std::string(hf_home_env) + "/hub";
}

std::string get_hf_model_path(std::string model_name) {
    auto slash = model_name.find_last_of('/');
    if(slash == std::string::npos) {
        throw std::runtime_error("HF model name must be of the form org/name");
    }
    model_name = "/models--" + model_name.replace(slash, 1, "--");
    return get_hf_hub() + model_name;
}

std::string get_hf_model_files(std::string model_name, std::string revision) {
    auto base_path = get_hf_model_path(model_name);
    if (!std::filesystem::exists(base_path))
        return "";

    if (revision.empty()) {
        std::string snapshot_path = base_path + "/snapshots/";
        for (auto& p : std::filesystem::directory_iterator(snapshot_path)) {
            if (!revision.empty())
                throw std::runtime_error("Found multiple snapshots, please specify a revision");
            if (p.is_directory())
                revision = p.path().filename();
        }
    }

    std::string revision_path = base_path + "/snapshots/" + revision;
    if (!std::filesystem::exists(revision_path))
        return "";
    return revision_path;
}
