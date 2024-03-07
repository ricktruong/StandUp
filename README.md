# StandUp
A simple AI-enhanced application for helping users maintain good posture in their working environment and improve ergonomics.

## Accredation
This project's OpenCV functionality was predominantly based on a tutorial put out by [LearnEDU](https://www.youtube.com/watch?v=70L3By4pci0&list=PLtsH5f9xHBUhiYoGs8_X6_R7P4IEMPrWn&index=1&t=3s&ab_channel=LearnEDU). This project's facial recognition software largely depended on the open-source [face_recognition library](https://github.com/ageitgey/face_recognition). Lastly, StandUp would like to extend our thanks to Professor Xihua Xiao and his team of instructional assistants for all the instruction, guidance, and mentorship along the way throughout the duration of this course.

## Setup
### Prerequisites
In order to successfully launch and run StandUp, the following requirements are needed:
- Python 3.3+
- macOS on Linux (is runnable but not officially supported on Windows)

In addition to the previous requirements, the following dependencies need to be installed:
- DLib
- NumPy
- OpenCV
- Homebrew
- cmake

### Running StandUp
#### Clone Source Code
In Terminal, clone this repository with the below command:
```bash
$ git clone https://github.com/ricktruong/StandUp.git
```

Change into the repository directory
```bash
$ cd StandUp
```

Run the main application Python script, and enjoy :)
```bash
python face_rec_timed.py
```
