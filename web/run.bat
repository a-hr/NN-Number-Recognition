@ECHO OFF
ECHO Running server...
START "" http://localhost:8000
cmd /k "python -m http.server 8000"
PAUSE