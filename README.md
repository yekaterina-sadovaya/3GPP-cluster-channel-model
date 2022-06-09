# 3GPP Spatial Channel Model (SCM)

## This repository provides an implementation of the SCM with a simple web GUI to run test scenarios.

### Description

This channel model is defined in the 3GPP TR 138.901 and suits for the frequency range from 0.5 up to 100 GHz.
The main purpose of this model is to provide an estimation of channel conditions between the transmitter and receiver.
The benefits of this model is that it provides not only power delay profile (PDP) but also angular profile.
The Python script with the model can be used as a standalone application. 
However, the purpose of GUI is to simplify the user expirience. The instructions on how to launch the test script are provided below.

### Instructions for running with GUI

- Clone this repository: https://github.com/yekaterina-sadovaya/3GPP-cluster-channel-model.git
- Run the HTTP server by launching the HTTP_server.py
- This will start the HTTP server at http://localhost:8000/
- Go to http://localhost:8000/ and click "Run Test Scenario"
- This will navigate you to the place where you can set the simulation parameters
- Set parameters according to the instruction provided on the web page and click "submit" button, which will send the AJAX request to the server 
- Wait for the results to appear on the same web page

![alt text](https://github.com/yekaterina-sadovaya/3GPP-cluster-channel-model/fig/example_results.png?raw=true)

### Instructions for running without GUI

Compute_channel.py can be used as a standalone file to compute the channel state. 
It takes the same input parameters as those mentioned on the web page.
More advanced parameters such as, e.g., delay spread, angular spread, K-factor, etc. can be changed inside the script if needed. 