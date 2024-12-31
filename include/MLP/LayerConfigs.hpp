#ifndef LAYER_CONFIGS_HPP
#define LAYER_CONFIGS_HPP

#ifndef EIGEN_USE_BLAS
#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACK
#endif

#include <Eigen/Dense>
#include "MLP/Optimizer.hpp"

namespace E=Eigen;
using E::MatrixXf;
using E::VectorXf;
using E::VectorXi;
using MatFunction = std::function<MatrixXf(const MatrixXf&)>;


enum LayerType{FeedForward,SoftMax,Convolutional};
/**
 * @brief Mostly for convolutional layers to translate vector
 */
struct InterfaceParams{
  int height;
  int width;
  int channels;
};

struct FFConfig{
  int feedforward_output;
};

struct CNNConfig{
  int cnn_kernel_number;
  int cnn_kernel_length;
};


/**
 * @brief For configuring the base layer
 *
 * Universal config struct and each type uses what is needs
 */
struct LayerConfig{
  LayerType layer_type;
  // For first layers
  int input_size;
  // For all layers
  int batch_size;
  MatFunction f,f_dot;
  // FF
  FFConfig ff_config;
  // CNN
  CNNConfig cnn_config;
  // For optimizer
  OptimizerMode optimizer_mode;
  SGDConfig sgd_config;
  AdamConfig adam_config;

};

#endif
