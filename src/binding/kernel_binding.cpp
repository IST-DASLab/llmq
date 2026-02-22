// Copyright (c) 2026, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "kernels/kernels.h"
#include "binding_utils.h"

#include "utilities/dtype.h"
#include "utilities/tensor.h"

namespace nb = nanobind;

using CudaArray = nb::ndarray<nb::c_contig, nb::device::cuda>;


Tensor to_tensor(const CudaArray& array) {
    const ETensorDType dtype = from_dlpack_dtype(array.dtype());
    const std::vector<long> shape(array.shape_ptr(), array.shape_ptr() + array.ndim());
    return Tensor::from_pointer(static_cast<std::byte*>(array.data()), array.device_id(), dtype, shape);
}
Tensor to_tensor(const std::optional<CudaArray>& array) {
    if (array.has_value()) {
        return to_tensor(array.value());
    } else {
        return Tensor{};
    }
}

long get_dimension(const std::initializer_list<std::size_t> values) {
    const long value = *values.begin();
    for (auto it = values.begin(); it != values.end(); ++it) {
        if (*it != value)
            throw std::invalid_argument("All dimensions must be equal");
    }
    return value;
}

void bind_encoder_forward(const CudaArray& out, const CudaArray& inp, const CudaArray& wte, const std::optional<CudaArray>& wpe, const std::uintptr_t stream) {
    // TODO wpe dimension check
    long B = get_dimension({out.shape(0), inp.shape(0)});
    long T = get_dimension({out.shape(1), inp.shape(1)});
    long V = wte.shape(0);
    long C = get_dimension({out.shape(2), wte.shape(1)});
    Tensor out_t = to_tensor(out);
    encoder_forward(out_t, to_tensor(inp), to_tensor(wte), to_tensor(wpe), B, T, C, V, reinterpret_cast<cudaStream_t>(stream));
}


void register_kernels(nanobind::module_& m) {
    m.def("encoder_forward", &bind_encoder_forward, nb::arg("out"), nb::arg("inp"), nb::arg("wte"), nb::arg("wpe") = std::nullopt, nb::arg("stream") = 0);
}
