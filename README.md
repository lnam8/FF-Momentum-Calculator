# Fantasy Football Momentum Calculator and Player Analyzer
With Docker installed, build an image using the supplied DockerFile.

In your terminal, run the code:

`docker build -t <image name> .`


Once that finishes, run the docker image using the command:


`docker run -p 8501:8501 <image name>`


This is not the typical docker run command, but it is imperative to be able to use the streamlit deployed app.

The terminal will tell you to view the streamlit app in your browser at `http://0.0.0.0:8501`, but entering `localhost:8501` in your browser will be sufficient.
