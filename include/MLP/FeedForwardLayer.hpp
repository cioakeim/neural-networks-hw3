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


  // For updating
  Optimizer weights_opt;
  Optimizer biases_opt;

public:
  
  FeedForwardLayer(){};

  // Initialization
  void configure(LayerConfig config) override;
  void init() override;

  // Forward (in case of manual input or of previous one's)
  void forward(const MatrixXf& input) override;
  void forward() override;

  float loss(const VectorXi& labels) override;
  int prediction_success(const VectorXi& labels) override{return -1;}

  // Backward (same idea as above)
  void backward(const MatrixXf& input,
                const VectorXi& labels) override;
  void backward(const MatrixXf& input) override;

  // For updating
  void updateWeights(const MatrixXf& input,
                     const MatrixXf& error);

  // I/O
  void store() override;
  void load() override;


};

#endif
