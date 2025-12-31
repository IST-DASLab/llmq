// Copyright (c) 2025, IST Austria, developed by Erik Schultheis
// SPDX-License-Identifier: Apache-2.0
//


#ifndef LLMQ_SRC_UTILS_TENSOR_CONTAINER_H
#define LLMQ_SRC_UTILS_TENSOR_CONTAINER_H

#include <functional>
#include <cstddef>
#include <string>

class Tensor;
class TensorShard;

class SimpleTensorContainer {
public:
    virtual std::size_t num_tensors() const noexcept = 0;
    virtual const Tensor& get_tensor(unsigned idx) const = 0;

    Tensor& get_tensor(unsigned idx) {
        return const_cast<Tensor&>(const_cast<const SimpleTensorContainer&>(*this).get_tensor(idx));
    }
protected:
    ~SimpleTensorContainer() = default;
};

//! \brief Apply the given function to all _non-empty_ tensors in `container`
void visit(const std::function<void(Tensor&)>& func, SimpleTensorContainer& container);

//! \brief Apply the given function to all _non-empty_ tensors in `a` and `b`.
//! \details `a` and `b` must have the same number of tensors, and empty tensors must be at the same indices
//! in both containers.
void visit(const std::function<void(Tensor&, Tensor&)>& func, SimpleTensorContainer& a, SimpleTensorContainer& b);

class ITensorContainer {
  public:
    virtual void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) = 0;

  protected:
    ~ITensorContainer() = default;
};

#endif //LLMQ_SRC_UTILS_TENSOR_CONTAINER_H
