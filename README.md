This repo stores MATLAB, Verilog, and C/C++ code for a Medical IoT Senior Design Project.
Associated Repos: [Need to include]
______________________________
MATLAB (Status: Training Model)
______________________________
Immediate Plans:
The MATLAB scripts created will perform the necessary work needed to collect data from an EEG database. 
The collected data will be processed through an FFT function and labeled before being trained with an SVM.
Once training is completed, the unused collected data will be used to validate the functionality of the binary
classifier.

Plans for a Future Version:
The inclusion of EEG data when a user is completing a simple exercise will be used 
as a way to gather baseline data to differentiate between normal signals and false positives.

______________________________
Verilog (Status: Design Phase)
______________________________
Immediate Plans:
A 256-point FFT will be created through hardware by employing a Hardware Description Language (HDL),
Verilog. This hardware addition will act as a hardware accelerator when processing EEG data that 
is coming in from our wearable device to our BeagleV-Fire

Future Plans:
N/A

______________________________
C/C++ (Status: In Progress)
______________________________
Immediate Plans:
Two C-based programs will be created: an inference model and a signal injection program.
Starting with the inference model, the collected parameters from the offline training will be used in an
inference model run on a BeagleV-Fire to run after signal data has been processed by the hardware accelerator.
The signal injection program, on the other hand, will be used to inject the collected data from the EEG database to validate
and demonstrate the capabilities of the device.

Future Plans:
For future use, it would be ideal to create a wrapper that would allow a user to more easily control the two C programs when needed

