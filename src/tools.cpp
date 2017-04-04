#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using namespace std;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if((estimations.size() != ground_truth.size()) || estimations.size() == 0){
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  //accumulate squared residuals
  for(unsigned int i=0; i < estimations.size(); ++i){

    VectorXd residual = estimations[i] - ground_truth[i];

    //coefficient-wise multiplication
    residual = residual.array()*residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse = rmse/estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const Eigen::VectorXd& x_state) {
  MatrixXd Hj(3,4);
  Hj << 0, 0, 0, 0, 
        0, 0, 0, 0, 
        0, 0, 0, 0;

  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  
  // Add a small number to prevent division by zero
  float denom = px*px + py*py + SMALL_NUMBER;
    
  // compute the Jacobian matrix 
  Hj(0, 0) = px/sqrt(denom);
  Hj(0, 1) = py/sqrt(denom);
  
  Hj(1, 0) = -py/denom;
  Hj(1, 1) = px/denom;
  
  Hj(2, 0) = py * (vx * py - vy * px)/(denom * sqrt(denom));
  Hj(2, 1) = px * (vy * px - vx * py)/(denom * sqrt(denom));
  Hj(2, 2) = px/sqrt(denom);
  Hj(2, 3) = py/sqrt(denom);

  return Hj;
}
