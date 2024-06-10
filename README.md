# action_detection_realtime_pytorch

## Description 
This project provides a workflow for training your own real-time action detection model using LSTM model. It includes a custom data generator designed to collect data directly from your camera, making it easy to create your own dataset for training.

## Reference Project 
This project is adapted from the tutorial provided by [nicknochnack](https://github.com/nicknochnack/ActionDetectionforSignLanguage). 

## Usage

### Installation
+ To install and set up the project, follow these steps:
	``` sh
	# Clone the repository
	git clone https://github.com/your-username/action-detection-lstm.git
	
	# Install the required dependencies
	pip install -r requirements.txt
	```

### Data
+ Captures frames from a video sequence and detects landmark using the Mediapipe lib, finally uses keypoints data for training.

+ Use the custom data generator to collect data from your camera:
	``` sh
	python action_data_generator.py your_action_labelname --output_dir output_directory --num_sequences number of sequence --frames_per_sequence number of frame
	```
	+ Example (example generated data  in 'data'):
		``` sh
		python action_data_generator.py nodding --output_dir 'data/' --num_sequences 100 --frames_per_sequence 20
		```
		
### Model Training and Testing
+ Use simple LSTM model for real-time prediction.
#### Training
+ Train your own model:
	``` sh
	python main.py data_directory run_name --result_dir output_directory --seqence_length length_of_seqence --epochs epochs_of_training
	```
	+ Example:
		``` sh
		python main.py 'data/' test --result_dir 'result/' --seqence_length 20 --epochs 200
		```

	+ This command outputs checkpoints, a log file (`logger.pt`), and plots of the accuracy and loss curves (if the optional `--save_plot` flag is used).
#### Testing
+ Test your own model in real-time:
	``` sh
	python realtime_action_dection.py checkpoint_directory
	```
	+ Example:
		``` sh
		python realtime_action_dection.py 'result/test_checkpoint.pt'
		```




