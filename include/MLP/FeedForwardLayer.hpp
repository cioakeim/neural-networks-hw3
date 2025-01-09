#ifndef FEEDFORWARD_LAYER_HPP
#define FEEDFORWARD_LAYER_HPP

#include "MLP/BaseLayer.hpp"
#include "MLP/Optimizer.hpp"
#include "MLP/BatchNormalization.hpp"


/**
 * @brief Feedforward Layer implementation
  */
class FeedForwardLayer:public BaseLayer{
protected:
  // Main components
  MatrixXf weights;
  MatrixXf biases;

  // Batch normalization flag
  bool batch_normalization;
  BatchNormHandler norm;

  // For updating
  Optimizer weights_opt;
  Optimizer biases_opt;
  Optimizer g_opt;
  Optimizer d_opt;

public:
  
  FeedForwardLayer(){};

  // Initialization
  void configure(LayerConfig config) override;
  void init() override;


  // Forward (in case of manual input or of previous one's)
  void forward(const PassContext& context) override;

  // Mean Square error of labels and prediction is on highest output
  float loss(const PassContext& context) override;
  int prediction_success(const PassContext& context) override;

  // Backward (same idea as above)
  // For Feedforward loss function is MSE
  void backward(const PassContext& context) override;

  void setLearningRate(const float rate) override{
    weights_opt.setRate(rate);biases_opt.setRate(rate);
    if(batch_normalization) norm.opt.setRate(rate);
  };

  // For updating
  void updateWeights(const MatrixXf& input,
                     const MatrixXf& error);

  // I/O
  void store() override;
  void load() override;

  void printStateInfo() override;

};

#endif
