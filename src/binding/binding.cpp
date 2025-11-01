#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/ndarray.h>

#include <filesystem>
#include <fmt/format.h>

#include "py_train.h"
#include "training/dataloader.h"
#include "training/checkpoint.h"
#include "training/logging.h"
#include "utilities/gpu_info.h"
#include "utilities/safetensors.h"
#include "utilities/sol.h"

namespace nb = nanobind;

using TokenArray = nb::ndarray<std::int32_t, nb::shape<-1, -1>, nb::device::cpu>;

static std::optional<ETensorDType> opt_dtype_from_str(const std::string& dtype_str) {
    if (dtype_str.empty()) {
        return std::nullopt;
    }
    return dtype_from_str(dtype_str);
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


NB_MODULE(_pyllmq, m) {
    nb::class_<GPUUtilInfo>(m, "GPUUtilInfo")
        .def_rw("clock", &GPUUtilInfo::clock)
        .def_rw("max_clock", &GPUUtilInfo::max_clock)
        .def_rw("power", &GPUUtilInfo::power)
        .def_rw("power_limit", &GPUUtilInfo::power_limit)
        .def_rw("fan", &GPUUtilInfo::fan)
        .def_rw("temperature", &GPUUtilInfo::temperature)
        .def_rw("temp_slowdown", &GPUUtilInfo::temp_slowdown)
        .def_rw("mem_free", &GPUUtilInfo::mem_free)
        .def_rw("mem_total", &GPUUtilInfo::mem_total)
        .def_rw("mem_reserved", &GPUUtilInfo::mem_reserved)
        .def_rw("gpu_utilization", &GPUUtilInfo::gpu_utilization)
        .def_rw("mem_utilization", &GPUUtilInfo::mem_utilization)
        .def_rw("throttle_reason", &GPUUtilInfo::throttle_reason)
        .def_rw("pcie_rx", &GPUUtilInfo::pcie_rx)
        .def_rw("pcie_tx", &GPUUtilInfo::pcie_tx)
        .def("__repr__", [](const GPUUtilInfo& gpu_util) {
            return fmt::format(
                R"(GPUUtilInfo(clock={}, max_clock={}, fan={}, power={}, power_limit={}, temperature={}, temp_slowdown={}, gpu_util={}, mem_util={}, throttle={}, dram_free={}, dram_total={}, dram_reserved={}, pcie_rx={}, pcie_tx={}))",
                                 gpu_util.clock, gpu_util.max_clock, gpu_util.fan, gpu_util.power, gpu_util.power_limit, gpu_util.temperature, gpu_util.temp_slowdown,
                                 gpu_util.gpu_utilization, gpu_util.mem_utilization, gpu_util.throttle_reason, gpu_util.mem_free, gpu_util.mem_total, gpu_util.mem_reserved,
                                 gpu_util.pcie_rx, gpu_util.pcie_tx);
        })
        ;

    nb::class_<LLamaConfig> (m, "LLamaConfig")
        .def("__init__", [](LLamaConfig *t,
            const std::string& arch, std::optional<int> bos_token_id, std::optional<int> eos_token_id,
            int hidden_size, int intermediate_size, std::optional<int> vocab_size, int num_attention_heads, int num_key_value_heads,
            int num_hidden_layers, std::optional<int> max_position_embeddings, std::optional<float> rope_theta, float rms_norm_eps, bool tie_word_embeddings, std::optional<bool> use_qkv_bias, std::string dtype) {
            // default values depend on selected architecture
             LLamaConfig::LLamaBasedModels architecture;
            if(arch == "qwen2" || arch == "Qwen2" || arch == "Qwen2ForCausalLM") {
                architecture = LLamaConfig::QWEN2;
                eos_token_id = eos_token_id.value_or(151643);
                bos_token_id = bos_token_id.value_or(151643);
                vocab_size = vocab_size.value_or(151936);
                max_position_embeddings = max_position_embeddings.value_or(32768);
                rope_theta = rope_theta.value_or(1000000.0);
                use_qkv_bias = use_qkv_bias.value_or(true);
            } else {
                throw std::runtime_error("At this point, only qwen2 architecture is supported.");
            }

            new (t) LLamaConfig{
                .Architecture = architecture,
                .BosTokenId = bos_token_id.value(),
                .EosTokenId = eos_token_id.value(),
                .HiddenSize = hidden_size,
                .IntermediateSize = intermediate_size,
                .VocabSize = vocab_size.value(),
                .NumQueryHeads = num_attention_heads,
                .NumKeyValHeads = num_key_value_heads,
                .NumLayers = num_hidden_layers,
                .MaxPositionEmbeddings = max_position_embeddings.value(),
                .RopeTheta = rope_theta.value(),
                .RmsNormEps = rms_norm_eps,
                .TiedWordEmbeddings = tie_word_embeddings,
                .UseQKVBias = use_qkv_bias.value(),
                .DType = dtype_from_str(dtype)
            };
        }, nb::kw_only(),
             nb::arg("architecture"), nb::arg("bos_token_id") = nb::none(), nb::arg("eos_token_id") = nb::none(), nb::arg("hidden_size"), nb::arg("intermediate_size"),
             nb::arg("vocab_size") = nb::none(), nb::arg("num_attention_heads"), nb::arg("num_key_value_heads"), nb::arg("num_hidden_layers"), nb::arg("max_position_embeddings") = nb::none(),
             nb::arg("rope_theta") = nb::none(), nb::arg("rms_norm_eps"), nb::arg("tie_word_embeddings"), nb::arg("use_qkv_bias") = nb::none(), nb::arg("dtype") = "bf16")
        .def_rw("architecture", &LLamaConfig::Architecture)
        .def_rw("bos_token_id", &LLamaConfig::BosTokenId)
        .def_rw("eos_token_id", &LLamaConfig::EosTokenId)
        .def_rw("hidden_size", &LLamaConfig::HiddenSize)
        .def_rw("intermediate_size", &LLamaConfig::IntermediateSize)
        .def_rw("vocab_size", &LLamaConfig::VocabSize)
        .def_rw("num_attention_heads", &LLamaConfig::NumQueryHeads)
        .def_rw("num_key_value_heads", &LLamaConfig::NumKeyValHeads)
        .def_rw("num_hidden_layers", &LLamaConfig::NumLayers)
        .def_rw("max_position_embeddings", &LLamaConfig::MaxPositionEmbeddings)
        .def_rw("rope_theta", &LLamaConfig::RopeTheta)
        .def_rw("rms_norm_eps", &LLamaConfig::RopeTheta)
        .def_rw("tie_word_embeddings", &LLamaConfig::TiedWordEmbeddings)
        .def_rw("use_qkv_bias", &LLamaConfig::UseQKVBias)
        .def_prop_rw("dtype",
                     [](const LLamaConfig* cfg){ return dtype_to_str(cfg->DType); },
                     [](LLamaConfig* cfg, const std::string& dtype_str){ cfg->DType = dtype_from_str(dtype_str); })
        .def_prop_ro("head_size", &LLamaConfig::head_size)
        .def_prop_ro("qkv_channels", &LLamaConfig::qkv_channels)
        .def_prop_ro("model_name", &LLamaConfig::model_name)
        .def_static("from_pretrained", [](const std::string& name, const std::string& dtype_str)
        {
            std::string hf_path = get_hf_model_files(name);
            if (hf_path.empty()) {
                throw std::runtime_error("Could not find model files for " + name);
            }
            std::string config_path = hf_path + "/config.json";
            return new LLamaConfig(load_llama_config(config_path.c_str(), dtype_from_str(dtype_str)));
        }, nb::arg("name"), nb::arg("dtype"), "Load the config file from an existing hf model")
        .def_static("from_name", [](const std::string& name, const std::string& dtype_str)
        {
            return new LLamaConfig(create_config_from_name(name, dtype_from_str(dtype_str)));
        }, nb::arg("name"), nb::arg("dtype"), "Create a config based on the model name.")
        ;

    nb::class_<LLamaOptions>(m, "LLamaOptions")
        .def("__init__", [](LLamaOptions *t, bool recompute_swiglu, bool recompute_rmsnorm,
            bool recompute_ffn, bool recompute_qkv, bool recompute_att, bool recompute_block,
            bool use_cuda_graphs,
            bool offload_master, bool offload_quants, bool offload_opt_m, bool offload_opt_v,
            bool use_write_combined, bool shard_weights, bool persistent_quants, bool shard_gradients, bool use_all_to_all_reduce, bool init_projections_to_zero,
            const std::string matmul_type, const std::string master_dtype, const std::string momentum_type, const std::string variance_type) {
            new (t) LLamaOptions{
                .RecomputeSwiGLu = recompute_swiglu,
                .RecomputeRMSNorm = recompute_rmsnorm,
                .RecomputeFFN = recompute_ffn,
                .RecomputeQKV = recompute_qkv,
                .RecomputeAtt = recompute_att,
                .RecomputeBlock = recompute_block,
                .UseCudaGraphs = use_cuda_graphs,
                .OffloadMaster = offload_master,
                .OffloadQuants = offload_quants,
                .OffloadOptM = offload_opt_m,
                .OffloadOptV = offload_opt_v,
                .UseWriteCombined = use_write_combined,
                .ShardWeights = shard_weights,
                .PersistentQuants = persistent_quants,
                .ShardGradients = shard_gradients,
                .UseAllToAllReduce = use_all_to_all_reduce,
                .InitProjectionsToZero = init_projections_to_zero,
                .MatmulType = opt_dtype_from_str(matmul_type),
                .MasterDType = opt_dtype_from_str(master_dtype),
                .OptMomentumType = dtype_from_str(momentum_type),
                .OptVarianceType = dtype_from_str(variance_type)
            };
        }, nb::kw_only(),
             nb::arg("recompute_swiglu") = false, nb::arg("recompute_rmsnorm") = false,
             nb::arg("recompute_ffn") = false,    nb::arg("recompute_qkv") = false,
             nb::arg("recompute_att") = false,    nb::arg("recompute_block") = false,
             nb::arg("use_cuda_graphs") = true,   nb::arg("offload_master") = false,
             nb::arg("offload_quants") = false,   nb::arg("offload_opt_m") = false,
             nb::arg("offload_opt_v") = false,    nb::arg("use_write_combined") = false,
             nb::arg("shard_weights") = false,    nb::arg("persistent_quants") = false,
             nb::arg("shard_gradients") = false,  nb::arg("use_all_to_all_reduce") = false,
             nb::arg("init_projections_to_zero") = false, nb::arg("matmul_type") = "",
             nb::arg("master_dtype") = "",        nb::arg("momentum_type") = "fp32",
             nb::arg("variance_type") = "fp32"
                 )
        .def_rw("recompute_swiglu", &LLamaOptions::RecomputeSwiGLu)
        .def_rw("recompute_rms_norm", &LLamaOptions::RecomputeRMSNorm)
        .def_rw("recompute_ffn", &LLamaOptions::RecomputeFFN)
        .def_rw("recompute_qkv", &LLamaOptions::RecomputeQKV)
        .def_rw("recompute_att", &LLamaOptions::RecomputeAtt)
        .def_rw("recompute_block", &LLamaOptions::RecomputeBlock)
        .def_rw("use_cuda_graphs", &LLamaOptions::UseCudaGraphs)
        .def_rw("offload_master", &LLamaOptions::OffloadMaster)
        .def_rw("offload_quants", &LLamaOptions::OffloadQuants)
        .def_rw("offload_opt_m", &LLamaOptions::OffloadOptM)
        .def_rw("offload_opt_v", &LLamaOptions::OffloadOptV)
        .def_rw("use_write_combined", &LLamaOptions::UseWriteCombined)
        .def_rw("shard_weights", &LLamaOptions::ShardWeights)
        .def_rw("persistent_quants", &LLamaOptions::PersistentQuants)
        .def_rw("shard_gradients", &LLamaOptions::ShardGradients)
        .def_rw("use_all_to_all_reduce", &LLamaOptions::UseAllToAllReduce)
        .def_rw("init_projections_to_zero", &LLamaOptions::InitProjectionsToZero)
        .def_prop_rw("matmul_type", [](const LLamaOptions* opt){ return dtype_to_str(opt->MatmulType.value()); },
                     [](LLamaOptions* opt, const std::string& dtype_str){ opt->MatmulType = dtype_from_str(dtype_str); })
        .def_prop_rw("master_dtype", [](const LLamaOptions* opt){ return dtype_to_str(opt->MasterDType.value()); },
                     [](LLamaOptions* opt, const std::string& dtype_str){ opt->MasterDType = dtype_from_str(dtype_str); })
        .def_prop_rw("momentum_type", [](const LLamaOptions* opt){ return dtype_to_str(opt->OptMomentumType); },
                     [](LLamaOptions* opt, const std::string& dtype_str){ opt->OptMomentumType = dtype_from_str(dtype_str); })
        .def_prop_rw("variance_type", [](const LLamaOptions* opt){ return dtype_to_str(opt->OptVarianceType); },
                     [](LLamaOptions* opt, const std::string& dtype_str){ opt->OptVarianceType = dtype_from_str(dtype_str); })
        ;

    nb::class_<MultiGPUPyTrainer>(m, "LLMQTrainer")
        .def("__init__", [](MultiGPUPyTrainer *t, int ngpu, LLamaConfig config, LLamaOptions options, int batch_size, int seq_len, int grad_accum, bool memcpy_all_gather, bool memcpy_send_recv) {
            new (t) MultiGPUPyTrainer(ngpu, config, options, batch_size, seq_len, grad_accum, memcpy_all_gather, memcpy_send_recv);
        }, nb::arg("ngpu"), nb::arg("config"), nb::arg("options"), nb::arg("batch_size"), nb::arg("seq_len"), nb::arg("grad_accum"),
             nb::arg("memcpy_all_gather") = true, nb::arg("memcpy_send_recv") = true)
        .def("import_weights", &MultiGPUPyTrainer::import_weights, nb::arg("path"), "Import weights from a hf model directory (model.safetensors)")
        .def("export_model", &MultiGPUPyTrainer::export_model, nb::arg("path"), "Export model as safetensors + config.json")
        .def_static("from_pretrained", [](const std::string& name, int ngpu, std::string dtype, LLamaOptions options, int batch_size, int seq_len, int grad_accum, bool memcpy_all_gather, bool memcpy_send_recv){
            std::string hf_path = get_hf_model_files(name);
            if (hf_path.empty()) {
                throw std::runtime_error("Could not find model files for " + name);
            }
            std::string config_path = hf_path + "/config.json";
            std::string model_path = hf_path + "/model.safetensors";
            if (!std::filesystem::exists(model_path)) {
                model_path = hf_path + "/model.safetensors.index.json";
            }
            LLamaConfig config = load_llama_config(config_path.c_str(), dtype_from_str(dtype));
            auto trainer = new MultiGPUPyTrainer(ngpu, config, options, batch_size, seq_len, grad_accum, memcpy_all_gather, memcpy_send_recv);
            trainer->import_weights(model_path);
            return trainer;
            }, nb::arg("name"), nb::arg("ngpu"), nb::arg("dtype"), nb::arg("options"), nb::arg("batch_size"), nb::arg("seq_len"), nb::arg("grad_accum"),
                    nb::arg("memcpy_all_gather") = true, nb::arg("memcpy_send_recv") = true, "Import weights from a hf model name")
        .def("init_weights", &MultiGPUPyTrainer::init_weights, "Initialize weights from scratch")
        .def("load_checkpoint", &MultiGPUPyTrainer::load_checkpoint, nb::arg("path"), nb::arg("step"), "Load a checkpoint for the specified step from the given checkpoint directory")
        .def("save_checkpoint", &MultiGPUPyTrainer::save_checkpoint, nb::arg("path"), nb::arg("step"), "Save a checkpoint for the specified step from to given checkpoint directory")
        .def("step", [](MultiGPUPyTrainer* trainer, TokenArray inputs, TokenArray targets) {
            CHECK_SHAPE(inputs, trainer->batch_size() * trainer->world_size(), trainer->seq_length());
            CHECK_SHAPE(targets, trainer->batch_size() * trainer->world_size(), trainer->seq_length());
            CHECK_CONTIGUOUS(inputs);
            CHECK_CONTIGUOUS(targets);

            trainer->step(inputs.data(), targets.data());
        }, nb::arg("inputs"), nb::arg("targets"),
             "Perform one step of forward/backward with the given inputs and targets. This function runs asynchronously; the loss will be made available during the next call to `update`.")
        .def("validate", [](MultiGPUPyTrainer* trainer, TokenArray inputs, TokenArray targets) {
            CHECK_SHAPE(inputs, trainer->batch_size() * trainer->world_size(), trainer->seq_length());
            CHECK_SHAPE(targets, trainer->batch_size() * trainer->world_size(), trainer->seq_length());
            CHECK_CONTIGUOUS(inputs);
            CHECK_CONTIGUOUS(targets);

            return trainer->validate(inputs.data(), targets.data());
        }, nb::arg("inputs"), nb::arg("targets"), "Perform one step of forward and loss calculation with the given inputs and targets, and return the resulting loss.")
        .def("update", [](MultiGPUPyTrainer* trainer, float lr, float beta1, float beta2, int step, float weight_decay, float grad_clip){
            auto [loss, norm] = trainer->update(lr, beta1, beta2, step, weight_decay, grad_clip);
            nb::dict ret;
            ret["loss"] = loss;
            ret["norm"] = norm;
            return ret;
        }, nb::arg("learning_rate"), nb::arg("beta1"), nb::arg("beta2"), nb::arg("step"), nb::arg("weight_decay"), nb::arg("grad_clip"),
             "Run the optimizer step and return the loss and gradient norm. This function blocks until the optimizer step is complete.")
        .def("get_gpu_info", &MultiGPUPyTrainer::get_gpu_info)
        .def_prop_ro("world_size", &MultiGPUPyTrainer::world_size)
        .def_prop_ro("batch_size", &MultiGPUPyTrainer::batch_size)
        .def_prop_ro("seq_length", &MultiGPUPyTrainer::seq_length)
        .def("get_allocator_info", [](MultiGPUPyTrainer* trainer, int gpu_id) {
            auto alloc = trainer->get_allocations(gpu_id);
            nb::dict ret;
            for(const auto& [name, size] : alloc) {
                ret[nb::cast(name)] = size;
            }
            return ret;
            }, nb::arg("gpu_id") = 0, "Get the current memory allocations for the given GPU")
        ;

    nb::class_<DataLoader>(m, "DataLoader")
        .def("__init__", [](DataLoader *d, const std::vector<std::string>& file_list, int chunk_size, unsigned long seed = 42) {
            new (d) DataLoader(file_list, chunk_size, 0, 1, seed);
        }, nb::arg("file_list"), nb::arg("chunk_size"), nb::arg("seed") = 42)
        .def("load_batch", [](DataLoader* d, TokenArray inputs, TokenArray targets) {
            CHECK_CONTIGUOUS(inputs);
            CHECK_CONTIGUOUS(targets);
            Tensor inp_t{ETensorDType::INT32, {static_cast<long>(inputs.shape(0)), static_cast<long>(inputs.shape(1)), 1, 1, 1},
                        reinterpret_cast<std::byte*>(inputs.data())};
            Tensor tgt_t{ETensorDType::INT32, {static_cast<long>(targets.shape(0)), static_cast<long>(targets.shape(1)), 1, 1, 1},
                        reinterpret_cast<std::byte*>(targets.data())};
            d->load_batch(inp_t, tgt_t);
        }, nb::arg("inputs"), nb::arg("targets"),
             "Fill inputs and targets with the next batch of data")
        .def("epoch", &DataLoader::epoch, "Get the current epoch number")
        .def("progress", &DataLoader::progress, "Get the current progress within the current epoch, in percent")
        .def("advance_epoch", &DataLoader::advance_epoch, "Advance to the next epoch, re-randomizing the order of chunks")
        .def("has_next", &DataLoader::has_next, nb::arg("chunks") = 1, "Check if there is another batch of data available")
        .def("set_state", &DataLoader::set_state, nb::arg("seed"), nb::arg("epoch"), nb::arg("file_index"), nb::arg("chunk_index"), "Sets the internal state of the dataloader.")
        .def_prop_ro("chunk_size", &DataLoader::chunk_size)
        .def_prop_ro("vocab_size", &DataLoader::vocab_size)
        .def_prop_ro("num_files", &DataLoader::num_files)
        .def_prop_ro("num_chunks", &DataLoader::num_chunks)
        .def_prop_ro("num_tokens", &DataLoader::num_tokens)
        .def_prop_ro("seed", &DataLoader::seed)
        ;

    m.def("find_latest_checkpoint", find_latest_checkpoint);
    m.def("get_all_checkpoints", get_all_checkpoints);
    m.def("get_checkpoint_path", get_checkpoint_path);
    m.def("find_latest_checkpoint", find_latest_checkpoint);
    m.def("clean_old_checkpoints", clean_old_checkpoints);
    m.def("get_num_gpus", [](){ int count; CUDA_CHECK(cudaGetDeviceCount(&count)); return count; });

    nb::enum_<TrainingRunLogger::EVerbosity>(m, "LogVerbosity")
        .value("SILENT", TrainingRunLogger::EVerbosity::SILENT)
        .value("QUIET", TrainingRunLogger::EVerbosity::QUIET)
        .value("DEFAULT", TrainingRunLogger::EVerbosity::DEFAULT)
        .value("VERBOSE", TrainingRunLogger::EVerbosity::VERBOSE)
        ;

    nb::class_<TrainingRunLogger>(m, "TrainingRunLogger", nb::dynamic_attr())
        .def("__init__", [](TrainingRunLogger *t, const std::string& file_name, nb::object callback_obj, TrainingRunLogger::EVerbosity verbosity) {
            new (t) TrainingRunLogger(file_name, 0, verbosity);
            if(!callback_obj.is_none()) {
                auto cb = nb::cast<nb::callable>(callback_obj);
                // set as an attribute on the python object to keep all ownership with python
                nb::setattr(nb::cast(t), "_callback", cb);
                t->set_callback([t](const std::string_view& msg) {
                    nb::gil_scoped_acquire gil;
                    auto cb = nb::cast<nb::callable>(nb::getattr(nb::cast(t), "_callback"));
                    cb(nb::cast(std::string(msg)));
                });
            }
        }, nb::arg("file_name"), nb::arg("callback") = nb::none(), nb::arg("verbosity") = TrainingRunLogger::EVerbosity::DEFAULT)
        .def("set_expected_time_per_token", &TrainingRunLogger::set_expected_time_per_token, nb::arg("nanoseconds"),
             "Set the expected time per token for MFU estimation")
        .def("log_cmd", [](TrainingRunLogger* logger, const std::vector<std::string>& args) {
            std::vector<const char*> argv;
            argv.reserve(args.size());
            for (const auto& arg : args) {
                argv.push_back(arg.c_str());
            }
            logger->log_cmd(args.size(), argv.data());
        }, nb::arg("args"), "Log command line arguments")

        .def("log_options", [](TrainingRunLogger* logger, const nb::dict& options) {
            std::vector<std::pair<std::string_view, std::variant<bool, int, float, std::string>>> cpp_options;
            std::vector<std::string> keys;
            keys.reserve(options.size());
            cpp_options.reserve(options.size());
            for (auto item : options) {
                nb::object value = nb::cast<nb::object>(item.second);
                keys.push_back(nb::cast<std::string>(item.first));
                std::string& key = keys.back(); // ensure key has sufficient lifetime
                if (nb::isinstance<nb::bool_>(value)) {
                    cpp_options.emplace_back(key, nb::cast<bool>(value));
                } else if (nb::isinstance<nb::int_>(value)) {
                    cpp_options.emplace_back(key, nb::cast<int>(value));
                } else if (nb::isinstance<nb::float_>(value)) {
                    cpp_options.emplace_back(key, nb::cast<float>(value));
                } else if (nb::isinstance<nb::str>(value)) {
                    cpp_options.emplace_back(key, nb::cast<std::string>(value));
                } else {
                    throw std::runtime_error("Unsupported option type for key: " + key);
                }
            }
            logger->log_options(cpp_options);
        }, nb::arg("options"), "Log training options as a dict")
        .def("log_dataset", &TrainingRunLogger::log_dataset, nb::arg("train_loader"), nb::arg("eval_loader"),
             "Log dataset information")
        .def("log_step", &TrainingRunLogger::log_step,
             nb::arg("step"), nb::arg("epoch"), nb::arg("step_tokens"), nb::arg("duration_ms"),
             nb::arg("norm"), nb::arg("loss"), nb::arg("lr"),
             "Log a training step")
        .def("log_eval", &TrainingRunLogger::log_eval,
             nb::arg("step"), nb::arg("epoch"), nb::arg("eval_tokens"), nb::arg("duration_ms"), nb::arg("loss"),
             "Log an evaluation step")
        .def("log_gpu_state", &TrainingRunLogger::log_gpu_state,
             nb::arg("step"), nb::arg("gpu_id"), nb::arg("gpu_util"),
             "Log GPU utilization state")
        .def("log_allocator", [](TrainingRunLogger* logger, const nb::dict& stats) {
            std::vector<std::pair<std::string, std::size_t>> cpp_stats;
            cpp_stats.reserve(stats.size());
            for (auto item : stats) {
                std::string key = nb::cast<std::string>(item.first);
                std::size_t value = nb::cast<std::size_t>(item.second);
                cpp_stats.emplace_back(key, value);
            }
            logger->log_allocator(cpp_stats);
        }, nb::arg("stats"), "Log memory allocator statistics")
        .def("log_checkpoint", &TrainingRunLogger::log_checkpoint,
             nb::arg("step"), nb::arg("path"), nb::arg("duration_ms"),
             "Log checkpoint save")
         .def("set_expected_time_per_token", [](TrainingRunLogger* logger, const MultiGPUPyTrainer* trainer){
             auto& config = trainer->config();
             auto& options = trainer->options();
                auto ops = get_transformer_ops(
                    config.NumLayers * ((long)config.HiddenSize * (config.IntermediateSize * 3 + config.HiddenSize * 1 + config.qkv_channels())),
                    options.MatmulType.value_or(config.DType), (long)config.VocabSize * config.HiddenSize, config.DType,
                    config.NumQueryHeads * config.head_size(), config.NumLayers, trainer->seq_length());
                 logger->set_expected_time_per_token(estimate_speed_of_light(get_gpu_name().c_str(), ops) / trainer->world_size());
             })
        ;
}
