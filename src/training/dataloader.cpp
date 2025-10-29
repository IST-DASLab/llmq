// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
//

#include "dataloader.h"

#include <glob.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <random>
#include <ranges>

#include <fmt/core.h>

#include "utilities/tensor.h"
#include "utilities/philox.h"

DataLoader::DataLoader(const std::string& file_pattern, int chunk_size, int rank, int world_size, unsigned long seed) :
       DataLoader(match_files(file_pattern), chunk_size, rank, world_size, seed) {

}

DataLoader::DataLoader(const std::vector<std::string>& file_list, int chunk_size, int rank, int world_size, unsigned long seed) :
        mChunkSize(chunk_size), mSeed(seed), mRank(rank), mWorldSize(world_size), mChunkIndex(rank) {
    if (file_list.empty()) {
        throw std::runtime_error("No token files provided");
    }

    for(const auto& file_name: file_list) {
        mFileInfos.push_back(parse_token_file_header(file_name));
    }

    mVocabSize = mFileInfos[0].VocabSize;
    for (auto& info: mFileInfos) {
        if (info.VocabSize != mVocabSize) {
            throw std::runtime_error(fmt::format("Inconsistent vocabulary sizes. Expected {}, got {} in {}.", mVocabSize, info.VocabSize, info.FileName));
        }
    }

    std::int64_t total_chunks = 0;
    std::int64_t total_tokens = 0;
    for (auto& info: mFileInfos) {
        total_tokens += info.NumTokens;
        total_chunks += (info.NumTokens - 1) / mChunkSize;
    }

    mTotalChunks = total_chunks;
    mTotalTokens = total_tokens;

    // this ensures that the first call to advance_epoch ends up in epoch 0
    mEpoch = -1;
    advance_epoch();
}

std::vector<std::string> DataLoader::match_files(const std::string& pattern) {
    std::vector<std::string> files;

    glob_t glob_result;
    int ret = glob(pattern.c_str(), GLOB_TILDE | GLOB_BRACE, nullptr, &glob_result);

    if (ret == 0 || ret == GLOB_NOMATCH) {
        for (size_t i = 0; i < glob_result.gl_pathc; ++i) {
            files.emplace_back(glob_result.gl_pathv[i]);
        }
        std::ranges::sort(files);
    } else {
        globfree(&glob_result);
        throw std::runtime_error(fmt::format("Failed to match files with pattern '{}': {}", pattern, glob_result.gl_pathv[0]));
    }

    globfree(&glob_result);
    return files;
}


DataLoader::TokenFileInfo DataLoader::parse_token_file_header(const std::string& file_name) {
    std::ifstream token_file(file_name, std::ios::binary);
    if (!token_file.is_open() || !token_file.good()) {
        throw std::runtime_error("Could not open token file: " + file_name);
    }
    token_file.exceptions(std::ifstream::failbit);

    TokenFileInfo info{.FileName = file_name};

    // read the header
    int header[256];
    token_file.read((char*)header, sizeof(header));
    constexpr char MAGIC[] = {'B', 'I', 'N', '.', 'T', 'O', 'K', '\n'};
    if(std::memcmp(header, MAGIC, sizeof(MAGIC)) != 0) {
        throw std::runtime_error(fmt::format("Invalid token file: '{}'", std::string_view((char*)header, sizeof(MAGIC))));
    }

    int version = header[2];
    if(version == 2) {
        info.VocabSize = header[5];
    } else if(version != 1) {
        throw std::runtime_error(fmt::format("Unsupported token file version: {}", version));
    }

    int bytes_per_token = header[3];
    if(bytes_per_token != 4) {
        throw std::runtime_error(fmt::format("Unsupported bytes per token: {}", bytes_per_token));
    }

    info.Version = version;
    info.BytesPerToken = bytes_per_token;
    info.NumTokens = header[4];
    return info;
}

float DataLoader::progress() const {
    std::int64_t epoch_tokens = 0;
    for (int i = 0; i < mFileIndex; ++i) {
        epoch_tokens += mShuffledFiles.at(i)->NumTokens;
    }
    epoch_tokens += mChunkIndex * mChunkSize;

    return 100.f * ((double)epoch_tokens / (double)mTotalTokens);
}

void DataLoader::shuffle_files() {
    mShuffledFiles.clear();
    for (auto& file : mFileInfos) {
        mShuffledFiles.push_back(&file);
    }
    Philox4x32 rng{mSeed};
    auto shuffle_seed = rng.generate(mEpoch, 0x73653753);
    std::ranges::shuffle(mShuffledFiles, std::default_random_engine{shuffle_seed[0]});
}

void DataLoader::shuffle_chunks() {
    int num_tokens = mShuffledFiles.at(mFileIndex)->NumTokens;
    int num_chunks = (num_tokens - 1) / mChunkSize;
    std::ranges::iota_view ids(0, num_chunks);
    mChunkOffsets.assign(std::begin(ids), std::end(ids));

    Philox4x32 rng{mSeed};
    auto shuffle_seed = rng.generate(mEpoch, mFileIndex);
    std::ranges::shuffle(mChunkOffsets, std::default_random_engine{shuffle_seed[0]});
}

bool DataLoader::advance_file() {
    ++mFileIndex;
    if (mFileIndex >= mShuffledFiles.size()) {
        return false;
    }

    // open the next file
    std::string file_name = mShuffledFiles.at(mFileIndex)->FileName;
    mTokenFile = std::ifstream(file_name, std::ios::binary);
    if (!mTokenFile.is_open() || !mTokenFile.good()) {
        throw std::runtime_error("Could not open token file: " + file_name);
    }
    mTokenFile.exceptions(std::ifstream::failbit);

    shuffle_chunks();

    // reset read position
    mChunkIndex = mRank;

    return true;
}

void DataLoader::advance_epoch() {
    ++mEpoch;
    shuffle_files();
    mFileIndex = -1;
    advance_file();
}

bool DataLoader::has_next(int n) const {
    if (mFileIndex != mShuffledFiles.size() - 1) {
        return true;
    }
    return mChunkIndex + n * mWorldSize - mRank < mChunkOffsets.size();
}

std::int32_t DataLoader::chunk_index() const {
    return mChunkIndex - mRank;
}

void DataLoader::load_batch(Tensor& inputs, Tensor& targets) {
    assert(inputs.Device == -1);
    assert(targets.Device == -1);
    if(inputs.nelem() != mChunkSize) {
        throw std::runtime_error(fmt::format("Expected inputs tensor of {} elements, got {}", mChunkSize, inputs.nelem()));
    }
    if(targets.nelem() != mChunkSize) {
        throw std::runtime_error(fmt::format("Expected targets tensor of {} elements, got {}", mChunkSize, targets.nelem()));
    }

    const long header_offset = 1024;
    const long element_size = 4;

    if (mChunkIndex + mWorldSize - mRank >= mChunkOffsets.size()) {
        if (!advance_file()) {
            throw std::runtime_error("No more files to load");
        }
    }

    try {
        const long input_bytes = inputs.bytes();
        const long target_bytes = targets.bytes();
        const int chunk_pos = mChunkSize * mChunkOffsets[mChunkIndex];
        const long input_offset = element_size * chunk_pos + header_offset;
        const long target_offset = input_offset + element_size;

        // Seek and read input data
        mTokenFile.seekg(input_offset, std::ios::beg);
        mTokenFile.read(reinterpret_cast<char*>(inputs.Data), input_bytes);

        // Verify we read the expected number of bytes
        if (mTokenFile.gcount() != static_cast<std::streamsize>(input_bytes)) {
            throw std::runtime_error("Incomplete read of input data: expected " +
                                     std::to_string(input_bytes) + " bytes, got " +
                                     std::to_string(mTokenFile.gcount()));
        }

        // Seek and read target data
        mTokenFile.seekg(target_offset, std::ios::beg);
        mTokenFile.read(reinterpret_cast<char*>(targets.Data), target_bytes);

        // Verify we read the expected number of bytes
        if (mTokenFile.gcount() != static_cast<std::streamsize>(target_bytes)) {
            throw std::runtime_error("Incomplete read of target data: expected " +
                                     std::to_string(target_bytes) + " bytes, got " +
                                     std::to_string(mTokenFile.gcount()));
        }

        // Update position only after successful reads
        mChunkIndex += mWorldSize;

    } catch (const std::ios_base::failure& e) {
        throw std::runtime_error("File I/O error: " + std::string(e.what()));
    }
}

void DataLoader::set_state(std::uint64_t seed, std::int32_t epoch, std::int32_t file_index, std::int32_t chunk_index) {
    mSeed = seed;
    mEpoch = epoch;
    mFileIndex = file_index;
    mChunkIndex = chunk_index + mRank;
}
