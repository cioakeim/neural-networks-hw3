#include "AutoEncoder/SSIMLayer.hpp"



float SSIMLayer::loss(const PassContext& context){
  const MatrixXf& output=output_interface->forward_signal;
  /*
  const MatrixXf& expected=context.input;
  // 2 first moments of prediction
  mx=output.colwise().mean();
  varx=
    2*(output.rowwise()-mx).colwise().sum()/(output.rows()-1);
  // 2 first moments of true output
  const MatrixXf my=expected.colwise().mean();

  //const MatrixXf vary=
  //  2*(expected.rowwise()-mx).colwise().sum()/(expected.rows()-1);
  // Covariance
  */
  return output.mean();  
}


MatrixXf SSIMLayer::initialError(const PassContext& context,
                                 const MatrixXf& output){
  return output;
}
