// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/optional.h>
#include <nanobind/ndarray.h>

#include <filesystem>
#include <fmt/format.h>

#include "py_train.h"
#include "binding_utils.h"
#include "training/dataloader.h"
#include "training/checkpoint.h"
#include "training/logging.h"
#include "utilities/gpu_info.h"
#include "utilities/safetensors.h"
#include "utilities/sol.h"

namespace nb = nanobind;

using TokenArray = nb::ndarray<std::int32_t, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;

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

#define CHECK_SHAPE(obj, ...) check_shape(obj, #obj, std::array{__VA_ARGS__})

void register_kernels(nanobind::module_& m);

NB_MODULE(_pyllmq, m) {
    register_kernels(m);

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

    nb::class_<TransformerConfig> (m, "Config")
        .def("__init__", [](TransformerConfig *t,
            const std::string& arch, std::optional<int> bos_token_id, std::optional<int> eos_token_id,
            int hidden_size, int intermediate_size, std::optional<int> vocab_size, int num_attention_heads, int num_key_value_heads,
            int num_hidden_layers, std::optional<int> max_position_embeddings, std::optional<float> rope_theta, float rms_norm_eps, bool tie_word_embeddings, std::optional<bool> use_qkv_bias, std::string dtype) {
            // default values depend on selected architecture
             TransformerConfig::EArchitecture architecture;
            if(arch == "qwen2" || arch == "Qwen2" || arch == "Qwen2ForCausalLM") {
                architecture = TransformerConfig::QWEN2;
                eos_token_id = eos_token_id.value_or(151643);
                bos_token_id = bos_token_id.value_or(151643);
                vocab_size = vocab_size.value_or(151936);
                max_position_embeddings = max_position_embeddings.value_or(32768);
                rope_theta = rope_theta.value_or(1000000.0);
                use_qkv_bias = use_qkv_bias.value_or(true);
            } else {
                throw std::runtime_error("At this point, only qwen2 architecture is supported.");
            }

            new (t) TransformerConfig{
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
        .def_rw("architecture", &TransformerConfig::Architecture)
        .def_rw("bos_token_id", &TransformerConfig::BosTokenId)
        .def_rw("eos_token_id", &TransformerConfig::EosTokenId)
        .def_rw("hidden_size", &TransformerConfig::HiddenSize)
        .def_rw("intermediate_size", &TransformerConfig::IntermediateSize)
        .def_rw("vocab_size", &TransformerConfig::VocabSize)
        .def_rw("num_attention_heads", &TransformerConfig::NumQueryHeads)
        .def_rw("num_key_value_heads", &TransformerConfig::NumKeyValHeads)
        .def_rw("num_hidden_layers", &TransformerConfig::NumLayers)
        .def_rw("max_position_embeddings", &TransformerConfig::MaxPositionEmbeddings)
        .def_rw("rope_theta", &TransformerConfig::RopeTheta)
        .def_rw("rms_norm_eps", &TransformerConfig::RopeTheta)
        .def_rw("tie_word_embeddings", &TransformerConfig::TiedWordEmbeddings)
        .def_rw("use_qkv_bias", &TransformerConfig::UseQKVBias)
        .def_prop_rw("dtype",
                     [](const TransformerConfig* cfg){ return dtype_to_str(cfg->DType); },
                     [](TransformerConfig* cfg, const std::string& dtype_str){ cfg->DType = dtype_from_str(dtype_str); })
        .def_prop_ro("head_size", &TransformerConfig::head_size)
        .def_prop_ro("qkv_channels", &TransformerConfig::qkv_channels)
        .def_prop_ro("model_name", &TransformerConfig::model_name)
        .def_static("from_pretrained", [](const std::string& name, const std::string& dtype_str)
        {
            std::string hf_path = get_hf_model_files(name);
            if (hf_path.empty()) {
                throw std::runtime_error("Could not find model files for " + name);
            }
            std::string config_path = hf_path + "/config.json";
            return new TransformerConfig(load_transformer_config(config_path.c_str(), dtype_from_str(dtype_str)));
        }, nb::arg("name"), nb::arg("dtype"), "Load the config file from an existing hf model")
        .def_static("from_name", [](const std::string& name, const std::string& dtype_str)
        {
            return new TransformerConfig(create_config_from_name(name, dtype_from_str(dtype_str)));
        }, nb::arg("name"), nb::arg("dtype"), "Create a config based on the model name.")
        ;

    nb::class_<LLamaOptions>(m, "LLamaOptions")
        .def("__init__", [](LLamaOptions *t, bool recompute_swiglu, bool recompute_rmsnorm,
            bool recompute_ffn, bool recompute_qkv, bool recompute_att, bool recompute_block, bool offload_residual,
            bool use_cuda_graphs,
            bool offload_master, bool offload_quants, bool offload_opt_m, bool offload_opt_v, bool offload_grads, bool use_zero_copy,
            bool use_write_combined, bool shard_weights, bool persistent_quants, bool shard_gradients, bool use_all_to_all_reduce,
            bool init_projections_to_zero, int lmhead_chunks, int attn_bwd_chunks,
            const std::string matmul_type, const std::string gradient_type, const std::string master_dtype, const std::string momentum_type, const std::string variance_type) {
            new (t) LLamaOptions{
                .RecomputeSwiGLu = recompute_swiglu,
                .RecomputeRMSNorm = recompute_rmsnorm,
                .RecomputeFFN = recompute_ffn,
                .RecomputeQKV = recompute_qkv,
                .RecomputeAtt = recompute_att,
                .RecomputeBlock = recompute_block,
                .OffloadResidual = offload_residual,
                .LMHeadChunks = lmhead_chunks,
                .AttBwdChunks = attn_bwd_chunks,
                .UseCudaGraphs = use_cuda_graphs,
                .OffloadMaster = offload_master,
                .OffloadQuants = offload_quants,
                .OffloadOptM = offload_opt_m,
                .OffloadOptV = offload_opt_v,
                .OffloadGrads = offload_grads,
                .UseZeroCopy = use_zero_copy,
                .UseWriteCombined = use_write_combined,
                .ShardWeights = shard_weights,
                .PersistentQuants = persistent_quants,
                .ShardGradients = shard_gradients,
                .UseAllToAllReduce = use_all_to_all_reduce,
                .InitProjectionsToZero = init_projections_to_zero,
                .MatmulType = opt_dtype_from_str(matmul_type),
                .GradientType = opt_dtype_from_str(gradient_type),
                .MasterDType = opt_dtype_from_str(master_dtype),
                .OptMomentumType = dtype_from_str(momentum_type),
                .OptVarianceType = dtype_from_str(variance_type)
            };
        }, nb::kw_only(),
             nb::arg("recompute_swiglu") = false, nb::arg("recompute_rmsnorm") = false,
             nb::arg("recompute_ffn") = false,    nb::arg("recompute_qkv") = false,
             nb::arg("recompute_att") = false,    nb::arg("recompute_block") = false,
             nb::arg("offload_residual") = false,
             nb::arg("use_cuda_graphs") = true,   nb::arg("offload_master") = false,
             nb::arg("offload_quants") = false,   nb::arg("offload_opt_m") = false,
             nb::arg("offload_opt_v") = false,    nb::arg("offload_grads") = false,
             nb::arg("use_zero_copy") = false,    nb::arg("use_write_combined") = false,
             nb::arg("shard_weights") = false,    nb::arg("persistent_quants") = false,
             nb::arg("shard_gradients") = false,  nb::arg("use_all_to_all_reduce") = false,
             nb::arg("init_projections_to_zero") = false, nb::arg("lmhead_chunks") = 1,
             nb::arg("attn_bwd_chunks") = 1,
             nb::arg("matmul_type") = "",         nb::arg("gradient_type") = "",
             nb::arg("master_dtype") = "",
             nb::arg("momentum_type") = "fp32",   nb::arg("variance_type") = "fp32"
                 )
        .def_rw("recompute_swiglu", &LLamaOptions::RecomputeSwiGLu)
        .def_rw("recompute_rms_norm", &LLamaOptions::RecomputeRMSNorm)
        .def_rw("recompute_ffn", &LLamaOptions::RecomputeFFN)
        .def_rw("recompute_qkv", &LLamaOptions::RecomputeQKV)
        .def_rw("recompute_att", &LLamaOptions::RecomputeAtt)
        .def_rw("recompute_block", &LLamaOptions::RecomputeBlock)
        .def_rw("offload_residual", &LLamaOptions::OffloadResidual)
        .def_rw("lmhead_chunks", &LLamaOptions::LMHeadChunks)
        .def_rw("attn_bwd_chunks", &LLamaOptions::AttBwdChunks)
        .def_rw("use_cuda_graphs", &LLamaOptions::UseCudaGraphs)
        .def_rw("offload_master", &LLamaOptions::OffloadMaster)
        .def_rw("offload_quants", &LLamaOptions::OffloadQuants)
        .def_rw("offload_opt_m", &LLamaOptions::OffloadOptM)
        .def_rw("offload_opt_v", &LLamaOptions::OffloadOptV)
        .def_rw("offload_grads", &LLamaOptions::OffloadGrads)
        .def_rw("use_zero_copy", &LLamaOptions::UseZeroCopy)
        .def_rw("use_write_combined", &LLamaOptions::UseWriteCombined)
        .def_rw("shard_weights", &LLamaOptions::ShardWeights)
        .def_rw("persistent_quants", &LLamaOptions::PersistentQuants)
        .def_rw("shard_gradients", &LLamaOptions::ShardGradients)
        .def_rw("use_all_to_all_reduce", &LLamaOptions::UseAllToAllReduce)
        .def_rw("init_projections_to_zero", &LLamaOptions::InitProjectionsToZero)
        .def_prop_rw("matmul_type", [](const LLamaOptions* opt){ return opt->matmul_dtype(); },
                     [](LLamaOptions* opt, const std::string& dtype_str){ opt->MatmulType = opt_dtype_from_str(dtype_str); })
        .def_prop_rw("gradient_type", [](const LLamaOptions* opt){ return opt->grad_dtype(); },
                     [](LLamaOptions* opt, const std::string& dtype_str){ opt->GradientType = opt_dtype_from_str(dtype_str); })
        .def_prop_rw("master_dtype", [](const LLamaOptions* opt){ return cast_opt_dtype(opt->MasterDType); },
                     [](LLamaOptions* opt, const std::string& dtype_str){ opt->MasterDType = opt_dtype_from_str(dtype_str); })
        .def_prop_rw("momentum_type", [](const LLamaOptions* opt){ return dtype_to_str(opt->OptMomentumType); },
                     [](LLamaOptions* opt, const std::string& dtype_str){ opt->OptMomentumType = dtype_from_str(dtype_str); })
        .def_prop_rw("variance_type", [](const LLamaOptions* opt){ return dtype_to_str(opt->OptVarianceType); },
                     [](LLamaOptions* opt, const std::string& dtype_str){ opt->OptVarianceType = dtype_from_str(dtype_str); })
        ;

    nb::class_<MultiGPUPyTrainer>(m, "LLMQTrainer")
        .def("__init__", [](MultiGPUPyTrainer *t, int ngpu, TransformerConfig config, LLamaOptions options, int batch_size, int seq_len, int grad_accum, bool memcpy_all_gather, bool memcpy_send_recv) {
            options.ModelType = config.DType;
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
            TransformerConfig config = load_transformer_config(config_path.c_str(), dtype_from_str(dtype));
            options.ModelType = config.DType;
            auto trainer = new MultiGPUPyTrainer(ngpu, config, options, batch_size, seq_len, grad_accum, memcpy_all_gather, memcpy_send_recv);
            trainer->import_weights(model_path);
            return trainer;
            }, nb::arg("name"), nb::arg("ngpu"), nb::arg("dtype"), nb::arg("options"), nb::arg("batch_size"), nb::arg("seq_len"), nb::arg("grad_accum"),
                    nb::arg("memcpy_all_gather") = true, nb::arg("memcpy_send_recv") = true, "Import weights from a hf model name")
        .def("init_weights", &MultiGPUPyTrainer::init_weights, "Initialize weights from scratch")
        .def("load_checkpoint", &MultiGPUPyTrainer::load_checkpoint, nb::arg("path"), nb::arg("step"), "Load a checkpoint for the specified step from the given checkpoint directory")
        .def("save_checkpoint", &MultiGPUPyTrainer::save_checkpoint, nb::arg("path"), nb::arg("step"), "Save a checkpoint for the specified step from to given checkpoint directory")
        .def("step", [](MultiGPUPyTrainer* trainer, TokenArray inputs, TokenArray targets, float z_loss) {
            CHECK_SHAPE(inputs, trainer->batch_size() * trainer->world_size(), trainer->seq_length());
            CHECK_SHAPE(targets, trainer->batch_size() * trainer->world_size(), trainer->seq_length());

            trainer->step(inputs.data(), targets.data(), z_loss);
        }, nb::arg("inputs"), nb::arg("targets"), nb::arg("z_loss") = 0.f,
             "Perform one step of forward/backward with the given inputs and targets. This function runs asynchronously; the loss will be made available during the next call to `update`.")
        .def("validate", [](MultiGPUPyTrainer* trainer, TokenArray inputs, TokenArray targets) {
            CHECK_SHAPE(inputs, trainer->batch_size() * trainer->world_size(), trainer->seq_length());
            CHECK_SHAPE(targets, trainer->batch_size() * trainer->world_size(), trainer->seq_length());

            return trainer->validate(inputs.data(), targets.data());
        }, nb::arg("inputs"), nb::arg("targets"), "Perform one step of forward and loss calculation with the given inputs and targets, and return the resulting loss.")
        .def("update", [](MultiGPUPyTrainer* trainer, float lr, float beta1, float beta2, int step, float weight_decay, float grad_clip){
            auto [loss, loss_1k, norm, z_max, z_mean] = trainer->update(lr, beta1, beta2, step, weight_decay, grad_clip);
            nb::dict ret;
            ret["loss"] = loss;
            ret["loss_1k"] = loss_1k;
            ret["norm"] = norm;
            ret["z_max"] = z_max;
            ret["z_mean"] = z_mean;
            return ret;
        }, nb::arg("learning_rate"), nb::arg("beta1"), nb::arg("beta2"), nb::arg("step"), nb::arg("weight_decay"), nb::arg("grad_clip"),
             "Run the optimizer step and return the loss and gradient norm. This function blocks until the optimizer step is complete.")
        .def("get_gradients", [](MultiGPUPyTrainer* trainer, int gpu_id)
        {
            auto raw = trainer->get_gradients(gpu_id);
            nb::dict ret;
            for (const auto& [name, value] : raw) {
                std::array<std::size_t, 6> shape;
                std::copy_n(value.Sizes.begin(), value.Rank, shape.begin());
                nb::ndarray<> view{value.Data, (size_t)value.Rank, shape.data(), ret, nullptr, to_dlpack_dtype(value.DType), nb::device::cuda::value, value.Device};
                ret[nb::cast(name)] = view;
            }
            return ret;
        }, "Return a dictionary with the model's gradient shards for the given GPU. This function is for debugging only: It is *blocking*!")
        .def("get_gpu_info", &MultiGPUPyTrainer::get_gpu_info)
        .def_prop_ro("world_size", &MultiGPUPyTrainer::world_size)
        .def_prop_ro("batch_size", &MultiGPUPyTrainer::batch_size)
        .def_prop_ro("seq_length", &MultiGPUPyTrainer::seq_length)
        .def("get_allocator_info", [](MultiGPUPyTrainer* trainer, int gpu_id) {
            auto alloc = trainer->get_allocations(gpu_id);
            nb::dict ret;
            for(const auto& [name, size] : alloc) {
                nb::dict res;
                res["device"] = size.OnDevice;
                res["managed"] = size.Managed;
                res["pinned"] = size.PinnedHost;
                res["pageable"] = size.PageableHost;
                ret[nb::cast(name)] = res;
            }

            auto stack = trainer->get_stack_info(gpu_id);
            for (const auto& [name, size] : stack) {
                nb::dict res;
                res["stack"] = size;
                ret[nb::cast(name)] = res;
            }
            return ret;
            }, nb::arg("gpu_id") = 0, "Get the current memory allocations for the given GPU")
        ;

    nb::class_<DataLoader>(m, "DataLoader")
        .def("__init__", [](DataLoader *d, const std::vector<std::string>& file_list, int chunk_size, unsigned long seed = 42) {
            new (d) DataLoader(file_list, chunk_size, 0, 1, seed);
        }, nb::arg("file_list"), nb::arg("chunk_size"), nb::arg("seed") = 42)
        .def("load_batch", [](DataLoader* d, TokenArray inputs, TokenArray targets) {
            Tensor inp_t{ETensorDType::INT32, {static_cast<long>(inputs.shape(0)), static_cast<long>(inputs.shape(1))},
                        reinterpret_cast<std::byte*>(inputs.data()), nullptr, 2, inputs.device_id()};
            Tensor tgt_t{ETensorDType::INT32, {static_cast<long>(targets.shape(0)), static_cast<long>(targets.shape(1))},
                        reinterpret_cast<std::byte*>(targets.data()), nullptr, 2, inputs.device_id()};
            d->load_batch(inp_t, tgt_t);
        }, nb::arg("inputs"), nb::arg("targets"),
             "Fill inputs and targets with the next batch of data")
        .def("epoch", &DataLoader::epoch, "Get the current epoch number")
        .def("progress", &DataLoader::progress, "Get the current progress within the current epoch, in percent")
        .def("advance_epoch", &DataLoader::advance_epoch, "Advance to the next epoch, re-randomizing the order of chunks")
        .def("has_next", &DataLoader::has_next, nb::arg("chunks") = 1, "Check if there is another batch of data available")
        .def("set_state", &DataLoader::set_state, nb::arg("seed"), nb::arg("epoch"), nb::arg("file_index"), nb::arg("chunk_index"), "Sets the internal state of the dataloader.")
        .def_prop_ro("seq_len", &DataLoader::seq_len)
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
        .def("log_cmd", [](TrainingRunLogger* logger, const std::vector<std::string>& args) {
            std::vector<const char*> argv;
            argv.reserve(args.size());
            for (const auto& arg : args) {
                argv.push_back(arg.c_str());
            }
            logger->log_cmd(args.size(), argv.data());
        }, nb::arg("args"), "Log command line arguments")

        .def("log_options", [](TrainingRunLogger* logger, const nb::dict& options) {
            std::vector<std::pair<std::string_view, std::variant<bool, long, float, std::string>>> cpp_options;
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
             nb::arg("step"), nb::arg("step_tokens"), nb::arg("duration_ms"),
             nb::arg("norm"), nb::arg("loss"), nb::arg("loss_1k"), nb::arg("logit_lse_max"), nb::arg("logit_lse_mean"), nb::arg("lr"),
             "Log a training step")
        .def("log_eval", &TrainingRunLogger::log_eval,
             nb::arg("step"), nb::arg("eval_tokens"), nb::arg("duration_ms"), nb::arg("loss"), nb::arg("loss_1k"),
             "Log an evaluation step")
        .def("log_gpu_state", &TrainingRunLogger::log_gpu_state,
             nb::arg("step"), nb::arg("gpu_id"), nb::arg("gpu_util"),
             "Log GPU utilization state")
        .def("log_allocator", [](TrainingRunLogger* logger, const nb::dict& stats) {
            std::vector<std::pair<std::string, sSegmentMemory>> cpp_stats;
            std::vector<std::pair<std::string, long>> cpp_stack;
            cpp_stats.reserve(stats.size());
            for (auto item : stats) {
                std::string key = nb::cast<std::string>(item.first);
                nb::dict value = nb::cast<nb::dict>(item.second);
                if (value.contains("stack")) {
                    cpp_stack.emplace_back(key, nb::cast<long>(value["stack"]));
                } else {
                    long device = nb::cast<long>(value["device"]);
                    long managed = nb::cast<long>(value["managed"]);
                    long pinned = nb::cast<long>(value["pinned"]);
                    long pageable = nb::cast<long>(value["pageable"]);
                    cpp_stats.emplace_back(key, sSegmentMemory{device, managed, pinned, pageable});
                }
            }
            logger->log_allocator(cpp_stats, cpp_stack);
        }, nb::arg("stats"), "Log memory allocator statistics")
         .def("set_expected_time_per_token", [](TrainingRunLogger* logger, const MultiGPUPyTrainer* trainer){
             auto& config = trainer->config();
             auto& options = trainer->options();
                auto ops = get_transformer_ops(
                    config.NumLayers * ((long)config.HiddenSize * (config.IntermediateSize * 3 + config.HiddenSize * 1 + config.qkv_channels())),
                    options.matmul_dtype(), (long)config.VocabSize * config.HiddenSize, config.DType,
                    config.NumQueryHeads * config.head_size(), config.NumLayers, trainer->seq_length());
                 logger->log_sol_estimate(ops, trainer->world_size());
             })
        ;
}
