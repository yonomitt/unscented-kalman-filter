#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  //
  // Calculate the RMSE here.
  //

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // estimations should not be zero length
  if (estimations.size() == 0) {
    cout << "ERROR - no estimations passed in" << endl;
    return rmse;
  }

  // estimations and ground_truth should be the same length
  if (estimations.size() != ground_truth.size()) {
    cout << "ERROR - estimation and ground truth are different sizes" << endl;
    return rmse;
  }

  // accumulate squared residuals
  for(int i = 0; i < estimations.size(); ++i) {
    VectorXd diff = estimations[i] - ground_truth[i];
    VectorXd diffSq = diff.array() * diff.array();

    rmse += diffSq;
  }

  // calculate the mean
  rmse /= estimations.size();

  // calculate the square root
  rmse = rmse.array().sqrt();

  // return the result
  return rmse;
}
