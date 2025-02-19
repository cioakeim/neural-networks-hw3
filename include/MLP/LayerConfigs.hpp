#ifndef LAYER_CONFIGS_HPP
#define LAYER_CONFIGS_HPP

#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACK
#endif

#include <Eigen/Dense>
#include <memory>
#include "MLP/Optimizer.hpp"

namespace E=Eigen;
using E::MatrixXf;
using E::VectorXf;
using E::VectorXi;
using MatFunction = std::function<MatrixXf(const MatrixXf&)>;


enum InterfaceType{Input,Hidden,Output};

/**
 * @brief Communication struct between 2 layers
 */
struct LayerInterface{
  // In/Out/Hidden
  InterfaceType type;
  // Dimensions of signals (tensors)
  int height;
  int width;
  int channels;
  // Forward and backward signals
  MatrixXf forward_signal;
  MatrixXf backward_signal;
  // Non-linearity is needed for both layers
  MatFunction f,f_dot;
};


struct FFConfig{
  int output_sz;
};

struct CNNConfig{
  int kernel_number;
  int kernel_dim;
};

enum LayerType{FeedForward,SoftMax,Convolutional,MSE};
/**
 * @brief What defines the properties of the black box
 */
struct LayerProperties{
  // The type of the layer
  LayerType layer_type;
  // For optimizer
  OptimizerConfig opt_config;
  // Batch normalization
  bool batch_normalization;
};



/**
 * @brief For configuring the base layer
 *
 * Universal config struct and each type uses what is needs
 */
struct LayerConfig{
  // The interface
  std::shared_ptr<LayerInterface> input_interface;
  std::shared_ptr<LayerInterface> output_interface;
  // The properties
  LayerProperties properties;
};


/**
 * @brief Context of a pass (forward and backward)
 */
struct PassContext{
  const MatrixXf input;
  const VectorXi labels;
};

#endif
