#ifndef AENC_CONFIG_HPP
#define AENC_CONFIG_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "MLP/LayerConfigs.hpp"
#include "MLP/Optimizer.hpp"

/**
 * @brief Config for general script info
  */
struct GeneralConfig{
  std::string dataset_path;
  std::string run_path;
  int training_size;
  int test_size;
  int batch_size;
  int epochs;
};

struct AutoEncoderConfig{
  std::vector<int> stack_sizes;
  std::vector<LayerType> stack_types;
  bool lock_weights;
  MatFunction f,f_dot; 
};

void configGeneral(GeneralConfig& config,
                   std::string config_filepath);

void configAutoEncoder(AutoEncoderConfig& config,
                       std::string config_filepath);

void configOptimizer(OptimizerConfig& config,
                     std::string config_filepath);

#endif
