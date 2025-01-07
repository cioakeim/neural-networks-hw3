#include "AutoEncoder/Config.hpp"
#include "MLP/ActivationFunctions.hpp"

void configGeneral(GeneralConfig& config,
                   std::string config_filepath){
  std::ifstream file(config_filepath);
  std::string line;
  // One by one standard assignment
  std::getline(file,line);
  config.dataset_path=line;
  std::getline(file,line);
  config.run_path=line;
  std::getline(file,line);
  config.training_size=std::stoi(line);
  std::getline(file,line);
  config.test_size=std::stoi(line);
  std::getline(file,line);
  config.batch_size=std::stoi(line);
  std::getline(file,line);
  config.epochs=std::stoi(line);
}

void configAutoEncoder(AutoEncoderConfig& config,
                       std::string config_filepath){
  std::ifstream file(config_filepath);
  std::string line;
  std::istringstream vecStream(line);
  std::string item;

  std::getline(file,line);
  vecStream=std::istringstream(line);
  while (std::getline(vecStream, item, ',')) {
      config.stack_sizes.push_back(std::stoi(item));
  }
  std::getline(file,line);
  vecStream=std::istringstream(line);
  while (std::getline(vecStream, item, ',')) {
    switch(item[0]){
      case 'M':
        config.stack_types.push_back(MSE);
      break;
      case 'F':
        config.stack_types.push_back(FeedForward);
      break;
      default:
        std::cerr<<"What layer is this??"<<std::endl;
        exit(1);
    }
  }
  std::getline(file,line);
  config.lock_weights=std::stoi(line);
  std::getline(file,line);
  if(line=="reLU"){
    config.f=reLU;config.f_dot=reLUder;
  } else if(line=="LreLU"){
    config.f=leakyReLU;config.f_dot=leakyReLUder;
  } else if(line=="linear"){
    config.f=linear;config.f_dot=linearder;
  }
  std::getline(file,line);

}

void configOptimizer(OptimizerConfig& config,
                     std::string config_filepath){
  std::ifstream file(config_filepath);
  std::string line;

  std::getline(file,line);
  if(line=="adam"){
    config.type=Adam;
    std::getline(file,line);
    config.adam.rate=std::stof(line);
    std::getline(file,line);
    config.adam.beta_1=std::stof(line);
    std::getline(file,line);
    config.adam.beta_2=std::stof(line);
  } 
  else if(line=="sgd"){
    config.type=SGD;
    std::getline(file,line);
    config.sgd.rate=std::stof(line);
  }
}
