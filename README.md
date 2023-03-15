# CV_Project_Basketball_Trajectory_with_Kalman
## Run Instructions
### 0. Install dependencies
`poetry install`
### 1. Run kalman_tracker.py
`python kalman_tracker.py --input {input_file} --model {model_type} --camshift --save-results --show`

Options:
- input: define the file to track
- model: choose between the available models (`nano, small, medium`) which to use to track the ball
- camshift: if present, once the ball is found, the script use camshift algorithm to track the ball
- save-results: if save the video result of the tracking process
- show: if the script has to show the tracking process real-time
