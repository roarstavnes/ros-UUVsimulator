# UUVsimulator for vision-aided inertial navigation using ArUco markers

> Link to the `uuv_simulator` repository [here](https://github.com/uuvsimulator/uuv_simulator)

> Link to the [documentation page](https://uuvsimulator.github.io/packages/uuv_simulator/intro/) 

> Link to the video of the simulation [youtube](https://www.youtube.com/watch?v=VIgVS2FLsy0)



# Quick start
The implementations are using ROS melodic and OpenCV.

```bash tab="lunar"
roslaunch uuv_gazebo_worlds ocean_waves.launch
```
```bash tab="lunar"
roslaunch rexrov2_description upload_rexrov2.launch
```
```bash tab="lunar"
roslaunch rexrov2_gazebo start_demo_pid_controller.launch teleop_on:=true joy_id:=0
```

Then open two terminals at the python folder and run
```bash tab="lunar"
python marker_detection_raw_ros.py
```
```bash tab="lunar"
python kf_attitude_ros.py
```


