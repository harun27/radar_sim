# radar_sim
This repository is about the simulation of multiple moving object tracking using multiple radars. 

Done: 
- Simulation of movement
- Simulation of radars and signal processing
- Localization of a single object
- Localization of multiple objects

Todo
- Investigate not only the reflexions but also the transmissions (think of a microphone array and how you can calculate the angle with the delay between the microphones)
- Plot the errors
- Implement Gauss-Newton Method to reduce the errors or WLLS (Weightes Linear Least Squares)
- Tracking:
  - Filtering (Kalman?)
  - Data Association
  - Track Management
- Use doppler effect and phase to simulataneouly measure distance and velocity
- Tackle NLOS (Non-Line-Of-Sight) issues
- Use RSS to mitigate the error further