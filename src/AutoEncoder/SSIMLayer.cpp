#include "AutoEncoder/SSIMLayer.hpp"



float SSIMLayer::loss(const PassContext& context){
  const MatrixXf& output=output_interface->forward_signal;
  const MatrixXf& expected=context.input;
  // 2 first moments of prediction
  mx=output.colwise().mean();
  varx=
    2*(output.rowwise()-mx.transpose()).colwise().sum()/(output.rows()-1);
  // 2 first moments of true output
  my=expected.colwise().mean();
  vary=
    2*(expected.rowwise()-my.transpose()).colwise().sum()/(expected.rows()-1);
  // Covariance
  cov=
  (output.rowwise()-mx.transpose()).cwiseProduct(
    expected.rowwise()-my.transpose()).colwise().mean();
  const float c1=0.01*0.01,c2=0.03*0.03;
  const VectorXf nom1=((2*mx.cwiseProduct(my)).array()+c1);
  const VectorXf nom2=(2*cov.array()+c2);
  const VectorXf den1=(mx.cwiseProduct(mx)+my.cwiseProduct(my)).array()+c1;
  const VectorXf den2=(varx.cwiseProduct(varx)+vary.cwiseProduct(vary)).array()+c2;
  nom=nom1.cwiseProduct(nom2);
  den=den1.cwiseProduct(den2);

  return 0.5*(nom.array()/den.array()).sum()+0.5*MSELayer::loss(context);
}


MatrixXf SSIMLayer::initialError(const PassContext& context,
                                 const MatrixXf& output){
  const MatrixXf& expected=context.input;
  const VectorXf covder=(1/output.rows()-1)*(expected.rowwise()-my);
  return output;
}
