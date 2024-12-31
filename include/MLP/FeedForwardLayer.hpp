#ifndef FEEDFORWARD_LAYER_HPP
#define FEEDFORWARD_LAYER_HPP

#include "MLP/BaseLayer.hpp"
#include "MLP/Optimizer.hpp"


/**
 * @brief Feedforward Layer implementation
  */
class FeedForwardLayer:public BaseLayer{
protected:
  // Main components
  MatrixXf weights;
  MatrixXf biases;
  // Changable parameters
  int batch_size;
  float rate;

  bool lockWeights=false;

  // For updating
  Optimizer weights_opt;
  Optimizer biases_opt;

public:
  
  FeedForwardLayer(){};

  // Initialization
  void configure(LayerConfig config) override;
  void init() override;

  // Locking 
  void lock(){lockWeights=true;}
  void unlock(){lockWeights=false;}

  // Forward (in case of manual input or of previous one's)
  void forward(const PassContext& context) override;

  float loss(const PassContext& context) override;
  int prediction_success(const PassContext& context) override{return -1;}

  // Backward (same idea as above)
  // For Feedforward loss function is MSE
  void backward(const PassContext& context) override;

  // For updating
  void updateWeights(const MatrixXf& input,
                     const MatrixXf& error);

  // I/O
  void store() override;
  void load() override;


};

#endif
