#ifndef SSIM_LAYER_HPP
#define SSIM_LAYER_HPP

#include "AutoEncoder/MSELayer.hpp"

class SSIMLayer: public MSELayer{
protected:
  MatrixXf mx;
  MatrixXf varx;
  float loss(const PassContext& context) override;

  MatrixXf initialError(const PassContext& context,
                        const MatrixXf& output) override;
};


#endif
