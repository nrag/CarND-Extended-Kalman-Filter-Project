#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <math.h> 

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
              0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
              0, 0.0009, 0,
              0, 0, 0.09;

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  //state covariance matrix P
  MatrixXd P_in = MatrixXd(4, 4);
  P_in << 1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1000, 0,
          0, 0, 0, 1000;


  //the initial transition matrix F_
  MatrixXd F_in = MatrixXd(4, 4);
  F_in << 1, 0, 1, 0,
          0, 1, 0, 1,
          0, 0, 1, 0,
          0, 0, 0, 1;

  MatrixXd Q_in = MatrixXd(4, 4);

  VectorXd x_in = VectorXd(4);
  x_in << 1, 1, 1, 1;

  ekf_.Init(x_in, P_in, F_in, H_laser_, R_laser_, Q_in);
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    // first measurement
    cout << "Initializing EKF: " << endl;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
       * Convert radar from polar to cartesian coordinates and initialize state.
       */
      float px = measurement_pack.raw_measurements_[0] * cos(measurement_pack.raw_measurements_[1]);
      float py = measurement_pack.raw_measurements_[0] * sin(measurement_pack.raw_measurements_[1]);
      float vx = measurement_pack.raw_measurements_[2] * cos(measurement_pack.raw_measurements_[1]);
      float vy = measurement_pack.raw_measurements_[2] * sin(measurement_pack.raw_measurements_[1]);

      //set the state with the initial location and zero velocity
      ekf_.x_ << px, py, 0, 0;
      cout << "Initialized x_ = " << ekf_.x_ << endl;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      //set the state with the initial location and zero velocity
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
      cout << "Initialized x_ = " << ekf_.x_ << endl;
    }

    // done initializing, no need to predict or update
    is_initialized_ = true;
    previous_timestamp_ = measurement_pack.timestamp_;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  //compute the time elapsed between the current and previous measurements
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  double dt_2 = dt * dt;
  double dt_3 = dt_2 * dt;
  double dt_4 = dt_3 * dt;
  
  //Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // Update the covariance matrix
  ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
              0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
              dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
              0, dt_3/2*noise_ay, 0, dt_2*noise_ay;
  
  //predict
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

      // Radar updates
      ekf_.R_ = R_radar_;
      Hj_ = tools.CalculateJacobian(ekf_.x_);
      ekf_.H_ = Hj_;
      ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }
  
  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
