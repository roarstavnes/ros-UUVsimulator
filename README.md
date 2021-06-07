# ros-UUVsimulator for vision-aided inertial navigation using ArUco markers

> Link to the `uuv_simulator` repository [here](https://github.com/uuvsimulator/uuv_simulator)

> Link to the [documentation page](https://uuvsimulator.github.io/packages/uuv_simulator/intro/) 

> Link to the video of the simulation [here](https://www.youtube.com/watch?v=VIgVS2FLsy0)



# Quick start
The implementations are using ROS melodic and OpenCV.

> roslaunch uuv_gazebo_worlds ocean_waves.launch

> roslaunch rexrov2_description upload_rexrov2.launch

> roslaunch rexrov2_gazebo start_demo_pid_controller.launch teleop_on:=true joy_id:=0

Then open two terminals at the python folder and run
> python marker_detection_raw_ros.py

> python kf_attitude_ros.py



