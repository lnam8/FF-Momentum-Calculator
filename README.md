# Fantasy Football Momentum Calculator and Player Analyzer
## Description
We developed an ensemble Machine Learning (ML) based player performance comparison model to determine the best players to start or sit in a team to win fantasy football games. Our team used a variety of techniques to compile a robust SQL Database from multiple sources, with which we performed ML analysis to make our decisions.


## Installation
Download the DockerFile and navigate to the directory it was downloaded to. Make sure the filename of the DockerFile is `dockerfile` and not `dockerfile.txt`. 

Open Docker locally.

Next, with Docker installed, build an image using the supplied DockerFile.

In your terminal, run the code:

`docker build -t <image name> .`

or

`docker build -t <image name> - < dockerfile.txt`


## Execution
Once the image is built, run the docker image using the command:

`docker run -p 8501:8501 <image name>`

This is not the typical docker run command, but it is imperative to be able to use the streamlit deployed app.

The terminal will tell you to view the streamlit app in your browser at `http://0.0.0.0:8501`. Enter this in your browser. If errors are received, enter `localhost:8501` into your browser.

Once opening `localhost:8501` in your browser, the app will load and you are free to explore our analysis for the available players.
