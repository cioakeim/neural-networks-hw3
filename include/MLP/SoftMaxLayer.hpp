#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "MLP/FeedForwardLayer.hpp"

/**
 * @brief Extension of Feedforward with softmax
*/
class SoftMaxLayer: public FeedForwardLayer{

  void forward(const PassContext& context) override;

  float loss(const PassContext& context) override;
  int prediction_success(const PassContext& context) override;

  void backward(const PassContext& context) override;

};

#endif // !SOFTMAX_HPP
