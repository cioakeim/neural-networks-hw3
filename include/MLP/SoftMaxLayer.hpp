#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "MLP/FeedForwardLayer.hpp"

/**
 * @brief Extension of Feedforward with softmax
*/
class SoftMaxLayer: public FeedForwardLayer{

  void forward(const MatrixXf& input) override;
  void forward() override;

  float loss(const VectorXi& labels) override;
  int prediction_success(const VectorXi& labels) override;

  void backward(const MatrixXf& input,
                const VectorXi& labels) override;

};

#endif // !SOFTMAX_HPP
