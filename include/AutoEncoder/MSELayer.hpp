#ifndef MSE_LAYER_HPP
#define MSE_LAYER_HPP

#include "MLP/FeedForwardLayer.hpp"

/**
 * @brief Extension of Feedforward with mse loss 
*/
class MSELayer : public FeedForwardLayer{
  void configure(LayerConfig config) override;

  void forward(const PassContext& context) override;

  float loss(const PassContext& context) override;
  int prediction_success(const PassContext& context) override;

  void backward(const PassContext& context) override;

  MatrixXf initialError(const PassContext& context,
                        const MatrixXf& output);

};


#endif
