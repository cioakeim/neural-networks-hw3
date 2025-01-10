#ifndef SSIM_LAYER_HPP
#define SSIM_LAYER_HPP

#include "AutoEncoder/MSELayer.hpp"

class SSIMLayer: public MSELayer{
protected:
  // Temp vars for no recalculating
  VectorXf mx;
  VectorXf varx;
  VectorXf my;
  VectorXf vary;
  VectorXf cov;
  VectorXf nom;
  VectorXf den;

  float loss(const PassContext& context) override;

  MatrixXf initialError(const PassContext& context,
                        const MatrixXf& output) override;
};


#endif
