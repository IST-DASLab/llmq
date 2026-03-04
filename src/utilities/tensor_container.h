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

//! \brief Base class for an object that supports iterating over its tensors
class SimpleTensorContainer {
public:
    //! Get the total number of tensors in this container. This count includes empty tensors.
    virtual std::size_t num_tensors() const noexcept = 0;

    //! Return a constant reference to the tensor at the given index.
    virtual const Tensor& get_tensor(std::size_t idx) const = 0;

    //! Return a mutable reference to the tensor at the given index.
    //! Implemented in terms of const `get_tensor`.
    Tensor& get_tensor(std::size_t idx) {
        return const_cast<Tensor&>(const_cast<const SimpleTensorContainer&>(*this).get_tensor(idx));
    }
protected:
    //! Protected destructor: Don't delete through `SimpleTensorContainer` pointers.
    ~SimpleTensorContainer() = default;
};

//! \brief Apply the given function to all _non-empty_ tensors in `container`
void visit(const std::function<void(Tensor&)>& func, SimpleTensorContainer& container);

//! \brief Apply the given function to all _non-empty_ tensors in `a` and `b`.
//! \details `a` and `b` must have the same number of tensors, and empty tensors must be at the same indices
//! in both containers.
void visit(const std::function<void(Tensor&, Tensor&)>& func, SimpleTensorContainer& a, SimpleTensorContainer& b);

//! \brief `SimpleTensorContainer` that stores all tensors in a vector
class GenericTensorContainer final : public SimpleTensorContainer {
public:
    GenericTensorContainer() = default;
    explicit GenericTensorContainer(std::vector<Tensor> t);

    //! Get the total number of tensors in this container. This count includes empty tensors.
    std::size_t num_tensors() const noexcept override;

    //! Return a constant reference to the tensor at the given index.
    const Tensor& get_tensor(std::size_t idx) const override;

    using SimpleTensorContainer::get_tensor;
private:
    std::vector<Tensor> mTensors;
};

//! Shards an _empty_ container, i.e., a container in which all data pointers
//! are `nullptr`, but sizes have been set up.
GenericTensorContainer shard_empty_container(GenericTensorContainer&& c, int world);

//! Shards a non-empty tensor container. The returned container's tensors are _views_ into
//! the original container's tensors.
GenericTensorContainer shard_view(const GenericTensorContainer& c, int rank, int world);

class ITensorContainer {
  public:
    virtual void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) = 0;

  protected:
    ~ITensorContainer() = default;
};

#endif //LLMQ_SRC_UTILS_TENSOR_CONTAINER_H
