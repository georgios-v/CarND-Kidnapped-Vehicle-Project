/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <vector>
#include <omp.h>

#include "map.h"
#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {

	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 101;
	particles.resize(num_particles);
	weights.resize(num_particles);

	std::normal_distribution<double> dist_x(0, std[0]);
	std::normal_distribution<double> dist_y(0, std[1]);
	std::normal_distribution<double> dist_t(0, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		particles[i] = Particle{i, x + dist_x(gen), y + dist_y(gen), theta + dist_t(gen), 1.0};
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::normal_distribution<double> dist_x(0, std_pos[0]);
	std::normal_distribution<double> dist_y(0, std_pos[1]);
	std::normal_distribution<double> dist_t(0, std_pos[2]);
#pragma omp parallel for
	for (uint i = 0; i < particles.size(); ++i) {
		Particle& p = particles[i];
		// calculate new state
		if (fabs(yaw_rate) < 0.00001) {
			p.x += delta_t * velocity * cos(p.theta);
			p.y += delta_t * velocity * sin(p.theta);
		} else {
			p.x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
			p.theta += yaw_rate * delta_t;
		}

		// add noise
		p.x += dist_x(gen);
		p.y += dist_y(gen);
		p.theta += dist_t(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<const Map::single_landmark_s*>& predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (uint i = 0; i < observations.size(); ++i) {
		LandmarkObs& obs = observations[i];
		double min_dist = std::numeric_limits<double>::max();
		for (uint j = 0; j < predicted.size(); ++j) {
			double d = dist(obs.x, obs.y, predicted[j]->x_f, predicted[j]->y_f);
			if (d < min_dist) {
				min_dist = d;
				obs.id = j;
			}
		}
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double sx = std_landmark[0];
	double sy = std_landmark[1];
	double norm = 2.0 * M_PI * sx * sy;
	double covx = 2.0 * sx * sx;
	double covy = 2.0 * sy * sy;

#pragma omp parallel for
	for (uint i = 0; i < particles.size(); ++i) {
		Particle& p = particles[i];

		std::vector<const Map::single_landmark_s*> inrange_landmarks;
		for (uint j = 0; j < map_landmarks.landmark_list.size(); ++j) {
			const Map::single_landmark_s& l = map_landmarks.landmark_list[j];
			if (std::fabs(l.x_f - p.x) <= sensor_range && std::fabs(l.y_f - p.y) <= sensor_range)
				inrange_landmarks.push_back(&l);
		}

		std::vector<LandmarkObs> observations_t(observations.size());
		for (uint j = 0; j < observations.size(); ++j) {
			const LandmarkObs& obs = observations[j];
			double x = obs.x * std::cos(p.theta) - obs.y * std::sin(p.theta) + p.x;
			double y = obs.x * std::sin(p.theta) + obs.y * std::cos(p.theta) + p.y;
			observations_t[j] = {obs.id, x, y};
		}

		dataAssociation(inrange_landmarks, observations_t);

		p.weight = 1.0;

		for (uint j = 0; j < observations_t.size(); ++j) {
			LandmarkObs& obs = observations_t[j];
			const Map::single_landmark_s *l = inrange_landmarks[obs.id];
			// Calculate multivariate Gaussian
			double divx = obs.x - l->x_f;
			double divy = obs.y - l->y_f;
			double w = exp(-(divx * divx / covx + divy * divy / covy)) / norm;
			// get total probability
			p.weight *= w;
		}
	}
}

static bool particle_compare(Particle& a, Particle& b) {
	return a.weight < b.weight;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	uint N = num_particles;
	double maxw = std::max_element(particles.begin(), particles.end(), &particle_compare)->weight;

	std::uniform_int_distribution<int> uidist(0, N - 1);
	std::uniform_real_distribution<double> urdist(0.0, maxw);
	int index = uidist(gen);
	double beta = 0.0;

	std::vector<Particle> p3;
	for (uint i = 0; i < N; ++i) {
		beta += urdist(gen) * 2.0;
		while (beta > particles[index].weight) {
			beta -= particles[index].weight;
			index = (index + 1) % N;
		}
		p3.push_back(particles[index]);
	}
	particles = p3;
}

Particle& ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
		const std::vector<double>& sense_x, const std::vector<double>& sense_y) {
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates
	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
	return particle;
}

std::string ParticleFilter::getAssociations(Particle best) {
	std::vector<int> v = best.associations;
	std::stringstream ss;
	copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
	std::string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}

std::string ParticleFilter::getSenseX(Particle best) {
	std::vector<double> v = best.sense_x;
	std::stringstream ss;
	copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
	std::string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}

std::string ParticleFilter::getSenseY(Particle best) {
	std::vector<double> v = best.sense_y;
	std::stringstream ss;
	copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
	std::string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
