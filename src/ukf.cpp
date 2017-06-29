#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


// Helper function to center the angle between -pi and pi
float centerAngle(float phi) {
  // if phi is too big, bring it down 2pi at a time
  while (phi > M_PI) {
    phi -= 2 * M_PI;
  }

  // if phi is too small, bring it up 2pi at a time
  while (phi < -M_PI) {
    phi += 2 * M_PI;
  }

  // return the new phi that is between -pi and pi
  return phi;
}


/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Set lambda
  lambda_ = 3 - n_aug_;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd::Zero(n_x_, 2 * n_aug_ + 1);

  // Weights of sigma points
  weights_ = VectorXd::Zero(2 * n_aug_ + 1);

  // time when the state is true, in us
  time_us_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  // initialization
  if (!is_initialized_) {
    
    // initialize the timestamp the the current measurement
    time_us_ = meas_package.timestamp_;

    // first position and velocity measurement
    x_ << 1, 1, 1, 1, 1;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state.
      float ro = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      float ro_dot = meas_package.raw_measurements_[2];

      x_[0] = ro * cos(phi);
      x_[1] = ro * sin(phi);
      x_[2] = ro_dot;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_[0] = meas_package.raw_measurements_[0];
      x_[1] = meas_package.raw_measurements_[1];
    }

    // initialize the covariance matrix
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;

    is_initialized_ = true;
    return;
  }

  // dt needs to be in seconds, but timestamp is in microseconds
  float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  Prediction(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    UpdateRadar(meas_package);
  } else {
    // Laser updates
    UpdateLidar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  //
  // Step 1: Generate augmented sigma points
  //

  // Create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // Create augmented state covariance
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);

  // Create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // Create augmented mean state
  x_aug << x_, 0, 0;

  // Create augmented covariance matrix
  P_aug.block(0, 0, n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_yawdd_ * std_yawdd_;

  // Create square root matrix
  MatrixXd A_aug = P_aug.llt().matrixL();

  // Create augmented sigma points
  Xsig_aug.col(0) = x_aug;

  MatrixXd X_aug = x_aug.rowwise().replicate(n_aug_);

  Xsig_aug.block(0, 1, n_aug_, n_aug_) = X_aug + sqrt(lambda_ + n_aug_) * A_aug;
  Xsig_aug.block(0, 1 + n_aug_, n_aug_, n_aug_) = X_aug - sqrt(lambda_ + n_aug_) * A_aug;

  //
  // Step 2: Predict sigma points
  //

  // Calcualte half of delta_t squared
  double half_dt2 = 0.5 * delta_t * delta_t;

  // Loop over each augmented sigma point
  for (int i = 0; i < Xsig_aug.cols(); i++)
  {
    VectorXd sigma_point = Xsig_aug.col(i);

    double px = sigma_point(0);
    double py = sigma_point(1);
    double v = sigma_point(2);
    double psi = sigma_point(3);
    double psi_dot = sigma_point(4);
    double nu_a = sigma_point(5);
    double nu_psi_dd = sigma_point(6);

    // Calculate the noise vector
    VectorXd noise = VectorXd(n_x_);
    noise << half_dt2 * cos(psi) * nu_a,
             half_dt2 * sin(psi) * nu_a,
             delta_t * nu_a,
             half_dt2 * nu_psi_dd,
             delta_t * nu_psi_dd;

    // Initialize the common part of the delta vector
    VectorXd delta_x = VectorXd(n_x_);
    delta_x << 0,
               0,
               0,
               psi_dot * delta_t,
               0;

    // Update the delta_x vector for the equations used to
    // avoid dividing by zero
    if (psi_dot == 0)
    {
      // simplified equations
      delta_x(0) = v * cos(psi) * delta_t;
      delta_x(1) = v * sin(psi) * delta_t;
    }
    else
    {
      // more complex update equations
      delta_x(0) = (v / psi_dot) * (sin(psi + psi_dot * delta_t) - sin(psi));
      delta_x(1) = (v / psi_dot) * (-cos(psi + psi_dot * delta_t) + cos(psi));
    }

    VectorXd x_k_plus_1 = sigma_point.head(n_x_) + delta_x + noise;
    
    Xsig_pred_.block(0, i, n_x_, 1) = x_k_plus_1;
  }

  //
  // Step 3: Predict mean and covariance
  //

  // Set weights vector
  int n_sigma = Xsig_pred_.cols();
  double l_plus_na = lambda_ + n_aug_;

  weights_(0) = lambda_ / l_plus_na;

  for (int i = 1; i < n_sigma; i++)
  {
    weights_(i) = 1 / (2 * l_plus_na);
  }

  // Predict state mean
  for (int i = 0; i < n_sigma; i++)
  {
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // Predict state covariance matrix
  for (int i = 0; i < n_sigma; i++)
  {
    VectorXd delta_x = Xsig_pred_.col(i) - x_;
    delta_x(3) = centerAngle(delta_x(3));
    P_ += weights_(i) * delta_x * delta_x.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  //
  // Step 1: Predict measurement
  //

  // Set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // Create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);

  // Measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  int n_sigma = Xsig_pred_.cols();

  // Transform sigma points into measurement space
  for (int i = 0; i < n_sigma; i++)
  {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double psi = Xsig_pred_(3, i);
    double psi_dot = Xsig_pred_(4, i);

    double ro = sqrt(px * px + py * py);
    double phi = (fabs(px) < 0.0001) ? 0.0 : atan2(py, px);
    double ro_dot = (fabs(ro) < 0.0001) ? 0.0 : (px * v * cos(psi) + py * v * sin(psi)) / ro;

    Zsig.col(i) << ro,
                   phi,
                   ro_dot;
  }

  // Calculate mean predicted measurement
  for (int i = 0; i < n_sigma; i++)
  {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // Calculate measurement covariance matrix S
  MatrixXd R = MatrixXd(n_z, n_z);
  R << std_radr_ * std_radr_, 0,                         0,
       0,                     std_radphi_ * std_radphi_, 0,
       0,                     0,                         std_radrd_ * std_radrd_;

  S = R;

  for (int i = 0; i < n_sigma; i++)
  {
    VectorXd delta_z = Zsig.col(i) - z_pred;
    delta_z(1) = centerAngle(delta_z(1));
    S += weights_(i) * delta_z * delta_z.transpose();
  }

  //
  // Step 2: Update state
  //

  // Set z to the raw measurements
  VectorXd z = meas_package.raw_measurements_;

  // Create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);

  // Calculate cross correlation matrix
  for (int i = 0; i < n_sigma; i++)
  {
    VectorXd delta_x = Xsig_pred_.col(i) - x_;
    delta_x(3) = centerAngle(delta_x(3));

    VectorXd delta_z = Zsig.col(i) - z_pred;
    delta_z(1) = centerAngle(delta_z(1));

    Tc += weights_(i) * delta_x * delta_z.transpose();
  }

  // Calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // Update state mean and covariance matrix
  VectorXd delta_z = z - z_pred;
  delta_z(1) = centerAngle(delta_z(1));
  x_ = x_ + K * delta_z;
  P_ = P_ - K * S * K.transpose();
}
