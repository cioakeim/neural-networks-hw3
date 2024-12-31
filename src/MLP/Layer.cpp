#include "MLP/Layer.hpp"
#include <fstream>
#include <random>

/**

Layer::Layer(int input_size,int output_size,int batch_size,float rate):
  weights(MatrixXf(output_size,input_size)),
  biases(VectorXf(output_size)),
  activations(MatrixXf(output_size,batch_size)),
  errors(MatrixXf(output_size,batch_size)){
};

Layer::Layer(std::string folder_path,const int batch_size){
  std::ifstream is;
  // Load weights
  is.open(folder_path+"/weights.csv",std::ios::in);
  int rows,cols;
  is>>rows>>cols;
  this->weights=E::MatrixXf(rows,cols);
  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++){
      is>>weights(i,j);
    }
  }
  is.close();
  // Load biases
  is.open(folder_path+"/biases.csv",std::ios::in);
  int size;
  is>>size;
  this->biases=E::VectorXf(size);
  for(int i=0;i<size;i++){
    is>>biases(i);
  }
  is.close();
  // Create the rest
  const int input_size=cols;
  const int layer_size=rows;
  this->errors=MatrixXf(layer_size,batch_size);
  this->activations=MatrixXf(layer_size,batch_size);
}

void Layer::updateWeights(const MatrixXf& input,
                          const float rate, const int batch_size){
  E::MatrixXf weightGradients=this->errors*(input.transpose());
  E::VectorXf biasGradients=this->errors.rowwise().sum();
  if(adam.epsilon>0){
    adam.update(weightGradients, biasGradients, weights, biases);
    return;
  }
  const float a=rate/batch_size;
  this->weights-=a*weightGradients+WEIGHT_DECAY*this->weights;
  this->biases-=a*biasGradients+WEIGHT_DECAY*this->biases;
}


// Store to location (2 files, 1 for weights and 1 for bias)
void Layer::store(std::string folder_path){
  std::ofstream os;
  // Store weights
  os.open(folder_path+"/weights.csv",std::ios::out);
  std::cout<<"Weights positive percentage: "<<(weights.array()>0).cast<float>().mean()<<std::endl;
  os<<weights.rows()<<" "<<weights.cols()<<"\n";
  for(int i=0;i<weights.rows();i++){
    for(int j=0;j<weights.cols();j++){
      os<<weights(i,j)<<" ";
    }
    os<<"\n";
  }
  os.close();
  // Store biases
  os.open(folder_path+"/biases.csv",std::ios::out);
  std::cout<<"Bias mean square: "<<(biases.array()>0).cast<float>().mean()<<std::endl;
  os<<biases.size()<<" "<<"\n"; 
  for(int i=0;i<biases.size();i++){
    os<<biases(i)<<"\n";
  }
  os.close();
}

// He Initialization accoutning for fan-in 
void Layer::HeRandomInit(){
  const int rows=weights.rows();
  const int cols=weights.cols();
  const float stddev= std::sqrt(2.0f/rows);
  // Init rng 
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0,stddev);

  for(int i=0;i<rows;i++){
    for(int j=0;j<cols;j++){
      weights(i,j)=(dist(gen));
    }
    biases(i)=(dist(gen));
  }
}


// Method with and without dropout
void Layer::activateDropout(const MatrixXf& input){
  this->activate(input);
  dropout.maskInput(activations);
}

void Layer::activateNormal(const MatrixXf& input){
  activations=f((weights*input).colwise()+biases);
}

// Softmax output
void Layer::softMaxForward(const MatrixXf& input){
  this->activate(input);
  const E::RowVectorXf maxCoeff=activations.colwise().maxCoeff();
  // Subtract for numerical stability and exp
  const MatrixXf exps=(activations.rowwise()-maxCoeff).array().exp();
  // Get sum of each column 
  const E::RowVectorXf col_sum=exps.colwise().sum();
  activations=exps.array().rowwise()/col_sum.array();
}

void Layer::softMaxBackward(const VectorXi& correct_labels){
  errors=activations;
  const int sample_size=activations.cols();
  #pragma omp parallel for
  for(int i=0;i<sample_size;i++){
    errors(correct_labels(i),i)--;
  }
}

// Back propagation
void Layer::activateErrors(const MatrixXf& next_weights,
                           const MatrixXf& next_errors){
  errors=(next_weights.transpose()*next_errors).cwiseProduct(f_dot(activations));
}





// Print methods
void Layer::printWeights(){
  // Simple print methods
  std::cout << this->weights << std::endl;
}

void Layer::printBiases(){
  std::cout << this->biases << std::endl;
}

void Layer::printActivations(){
  std::cout << this->activations<< std::endl;
}

void Layer::printErrors(){
  std::cout << this->errors<< std::endl;
}
*/
