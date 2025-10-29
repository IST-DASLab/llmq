#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/ndarray.h>

#include <fmt/format.h>

#include "py_train.h"
#include "training/dataloader.h"
#include "training/checkpoint.h"

namespace nb = nanobind;

using TokenArray = nb::ndarray<std::int32_t, nb::shape<-1, -1>, nb::device::cpu>;

static std::optional<ETensorDType> opt_dtype_from_str(const std::string& dtype_str) {
    if (dtype_str.empty()) {
        return std::nullopt;
    }
    return dtype_from_str(dtype_str);
}

static bool get_bool_from_dict(nb::dict dict_obj, const char* key, bool default_val) {
    return nb::cast<bool>(dict_obj.get(key, nb::cast(default_val)));
}

static std::string get_string_from_dict(nb::dict dict_obj, const char* key, const char* default_val = "") {
    return nb::cast<std::string>(dict_obj.get(key, nb::cast(default_val)));
}

static int get_int_from_dict(nb::dict dict_obj, const char* key, int default_val = 0) {
    return nb::cast<int>(dict_obj.get(key, nb::cast(default_val)));
}

static float get_float_from_dict(nb::dict dict_obj, const char* key, float default_val = 0.0f) {
    return nb::cast<float>(dict_obj.get(key, nb::cast(default_val)));
}

LLamaConfig config_from_dict(nb::dict dict_obj) {
    LLamaConfig config;

    // Architecture enum (string -> enum mapping)
    std::string arch_str = get_string_from_dict(dict_obj, "architecture");
    if (arch_str == "qwen2" || arch_str == "QWEN2") {
        config.Architecture = LLamaConfig::QWEN2;
    } else {
        config.Architecture = LLamaConfig::LLAMA;  // Default to LLAMA
    }

    // Integer fields
    config.BosTokenId = get_int_from_dict(dict_obj, "bos_token_id", 1);
    config.EosTokenId = get_int_from_dict(dict_obj, "eos_token_id", 2);
    config.HiddenSize = get_int_from_dict(dict_obj, "hidden_size");
    config.IntermediateSize = get_int_from_dict(dict_obj, "intermediate_size");
    config.VocabSize = get_int_from_dict(dict_obj, "vocab_size");
    config.NumQueryHeads = get_int_from_dict(dict_obj, "num_attention_heads");
    config.NumKeyValHeads = get_int_from_dict(dict_obj, "num_key_value_heads");
    config.NumLayers = get_int_from_dict(dict_obj, "num_hidden_layers");
    config.MaxPositionEmbeddings = get_int_from_dict(dict_obj, "max_position_embeddings");

    // Float fields
    config.RopeTheta = get_float_from_dict(dict_obj, "rope_theta", 10000.0f);
    config.RmsNormEps = get_float_from_dict(dict_obj, "rms_norm_eps", 1e-6f);

    // Boolean fields
    config.TiedWordEmbeddings = get_bool_from_dict(dict_obj, "tie_word_embeddings", false);
    config.UseQKVBias = get_bool_from_dict(dict_obj, "use_qkv_bias", false);

    // DType field
    std::string dtype_str = get_string_from_dict(dict_obj, "dtype");
    if (!dtype_str.empty()) {
        config.DType = dtype_from_str(dtype_str);
    }

    return config;
}

LLamaOptions options_from_dict(nb::dict dict_obj) {
    auto get_bool = [dict_obj](const char* key, bool default_val) -> bool {
        return get_bool_from_dict(dict_obj, key, default_val);
    };

    auto get_string = [dict_obj](const char* key) -> std::string {
        return get_string_from_dict(dict_obj, key);
    };

    LLamaOptions options;

    // Boolean options
    options.KeepAllActivations = get_bool("keep_all_activations", false);
    options.RecomputeSwiGLu = get_bool("recompute_swiglu", false);
    options.RecomputeRMSNorm = get_bool("recompute_rmsnorm", false);
    options.RecomputeFFN = get_bool("recompute_ffn", false);
    options.RecomputeQKV = get_bool("recompute_qkv", false);
    options.RecomputeAtt = get_bool("recompute_att", false);
    options.RecomputeBlock = get_bool("recompute_block", false);
    options.UseCudaGraphs = get_bool("use_cuda_graphs", false);

    options.OffloadMaster = get_bool("offload_master", false);
    options.OffloadQuants = get_bool("offload_quants", false);
    options.OffloadOptM = get_bool("offload_opt_m", false);
    options.OffloadOptV = get_bool("offload_opt_v", false);
    options.UseWriteCombined = get_bool("use_write_combined", false);
    options.ShardWeights = get_bool("shard_weights", false);
    options.PersistentQuants = get_bool("persistent_quants", false);

    options.ShardGradients = get_bool("shard_gradients", false);
    options.UseAllToAllReduce = get_bool("use_all_to_all_reduce", false);

    options.InitProjectionsToZero = get_bool("init_projections_to_zero", false);

    // Optional dtype options
    options.MatmulType = opt_dtype_from_str(get_string("matmul_type"));
    options.MasterDType = opt_dtype_from_str(get_string("master_dtype"));

    // Required dtype options with defaults
    if (auto opt = opt_dtype_from_str(get_string("opt_momentum_type")); opt.has_value()) {
        options.OptMomentumType = opt.value();
    }

    if (auto opt = opt_dtype_from_str(get_string("opt_variance_type")); opt.has_value()) {
        options.OptVarianceType = opt.value();
    }

    return options;
}

template<typename NBArray, std::size_t NDims>
static inline auto check_shape(const NBArray& arr, std::string_view name, std::array<int, NDims> expected) {
    if(arr.ndim() != expected.size()) {
        throw std::runtime_error(fmt::format("Expected {} to have {} dimensions, but got {}", name, expected.size(), arr.ndim()));
    }
    for(int dim = 0; dim < expected.size(); ++dim) {
        if (arr.shape(dim) != expected[dim]) {
            throw std::runtime_error(
                fmt::format("Expected {} to have extent {} at dimension {}, but got {}", name, expected[dim], dim,
                            arr.shape(dim)));
        }
    }
}

template<typename NBArray>
static inline auto check_contiguous(const NBArray& arr, std::string_view name) {
    int stride = 1;
    for(int dim = arr.ndim() - 1; dim >= 0; --dim) {
        if(arr.stride(dim) != stride) {
            throw std::runtime_error(fmt::format("Expected {} to be contiguous", name));
        }
        stride *= arr.shape(dim);
    }
}

#define CHECK_SHAPE(obj, ...) check_shape(obj, #obj, std::array{__VA_ARGS__})
#define CHECK_CONTIGUOUS(obj) check_contiguous(obj, #obj)


NB_MODULE(pyllmq, m) {
    nb::class_<MultiGPUPyTrainer>(m, "LLMQTrainer")
        .def("__init__", [](MultiGPUPyTrainer *t, int ngpu, nb::dict config, nb::dict options, int batch_size, int seq_len, int grad_accum, bool memcpy_all_gather, bool memcpy_send_recv) {
            new (t) MultiGPUPyTrainer(ngpu, config_from_dict(config), options_from_dict(options), batch_size, seq_len, grad_accum, memcpy_all_gather, memcpy_send_recv);
        })
        .def("import_weights", &MultiGPUPyTrainer::import_weights)
        .def("init_weights", &MultiGPUPyTrainer::init_weights)
        .def("load_checkpoint", &MultiGPUPyTrainer::load_checkpoint)
        .def("save_checkpoint", &MultiGPUPyTrainer::save_checkpoint)
        .def("step", [](MultiGPUPyTrainer* trainer, TokenArray inputs, TokenArray targets) {
            CHECK_SHAPE(inputs, trainer->batch_size() * trainer->world_size(), trainer->seq_length());
            CHECK_SHAPE(targets, trainer->batch_size() * trainer->world_size(), trainer->seq_length());
            CHECK_CONTIGUOUS(inputs);
            CHECK_CONTIGUOUS(targets);

            trainer->step(inputs.data(), targets.data());
        })
        .def("validate", [](MultiGPUPyTrainer* trainer, TokenArray inputs, TokenArray targets) {
            CHECK_SHAPE(inputs, trainer->batch_size() * trainer->world_size(), trainer->seq_length());
            CHECK_SHAPE(targets, trainer->batch_size() * trainer->world_size(), trainer->seq_length());
            CHECK_CONTIGUOUS(inputs);
            CHECK_CONTIGUOUS(targets);

            return trainer->validate(inputs.data(), targets.data());
        })
        .def("update", [](MultiGPUPyTrainer* trainer, float lr, float beta1, float beta2, int step, float weight_decay, float grad_clip){
            auto [loss, norm] = trainer->update(lr, beta1, beta2, step, weight_decay, grad_clip);
            nb::dict ret;
            ret["loss"] = loss;
            ret["norm"] = norm;
            return ret;
        })
        .def("stop", &MultiGPUPyTrainer::stop);

    nb::class_<DataLoader>(m, "DataLoader")
        .def("__init__", [](DataLoader *d, const std::vector<std::string>& file_list, int chunk_size, unsigned long seed = 42) {
            new (d) DataLoader(file_list, chunk_size, 0, 1, seed);
        })
        .def("load_batch", [](DataLoader* d, TokenArray inputs, TokenArray targets) {
            CHECK_CONTIGUOUS(inputs);
            CHECK_CONTIGUOUS(targets);
            Tensor inp_t{ETensorDType::INT32, {static_cast<long>(inputs.shape(0)), static_cast<long>(inputs.shape(1)), 1, 1, 1},
                        reinterpret_cast<std::byte*>(inputs.data())};
            Tensor tgt_t{ETensorDType::INT32, {static_cast<long>(targets.shape(0)), static_cast<long>(targets.shape(1)), 1, 1, 1},
                        reinterpret_cast<std::byte*>(targets.data())};
            d->load_batch(inp_t, tgt_t);
        })
        .def("epoch", &DataLoader::epoch)
        .def("progress", &DataLoader::progress)
        .def("advance_epoch", &DataLoader::advance_epoch)
        .def("has_next", &DataLoader::has_next)
        .def_prop_ro("chunk_size", &DataLoader::chunk_size)
        .def_prop_ro("vocab_size", &DataLoader::vocab_size)
        .def_prop_ro("num_files", &DataLoader::num_files)
        .def_prop_ro("num_chunks", &DataLoader::num_chunks)
        .def_prop_ro("num_tokens", &DataLoader::num_tokens)
        ;

    m.def("find_latest_checkpoint", find_latest_checkpoint);
    m.def("get_all_checkpoints", get_all_checkpoints);
    m.def("get_checkpoint_path", get_checkpoint_path);
}
